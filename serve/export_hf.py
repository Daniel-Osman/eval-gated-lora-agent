"""Publish the served artifact and the LoRA adapter to a private Hugging Face repo.

The weights live on the Modal Volume, so the upload runs in a Modal container:
17.3 GiB never crosses this laptop and the Hub token never leaves the secret.

Three functions, one job each:

    probe()   what the token is, who it belongs to, and what is on the Volume.
              Uploads nothing.
    export()  create the repo private, fail closed if it is not, upload, write the
              model card, return the file listing and the commit ids.
    verify()  read the repo back with the same token: the private flag, every file
              with its size, an anonymous refusal, and the two weight sizes checked
              against the bytes the Volume holds.

    python -m modal run serve/export_hf.py --probe-only true
    python -m modal run serve/export_hf.py --private true
    python -m modal run serve/export_hf.py --verify-only <account>/<repo>
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import modal

VOLUME_NAME = "eval-gated-lora-agent"
VOLUME_MOUNT = "/vol"
QUANT_DIR = f"{VOLUME_MOUNT}/serve/slice1/awq-w4a16"
ADAPTER_DIR = f"{VOLUME_MOUNT}/runs/full-20261114T201900Z/adapter"

SECRET_NAME = "scenesift-hf"
REPO_NAME = "eval-gated-qwen3.8-27b"

# The two files whose sizes the record fixed. `verify` fails if the Hub reports
# anything else, so a truncated upload cannot pass as success.
QUANT_WEIGHTS_BYTES = 18_568_390_152
ADAPTER_WEIGHTS_BYTES = 467_062_560

# Both names are accepted by huggingface_hub; the second is its legacy alias.
TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")

MODEL_CARD = '''---
license: apache-2.0
base_model: unsloth/Qwen3.8-27B-unsloth-bnb-4bit
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen3
  - awq
  - w4a16
  - 4-bit
  - vllm
  - qlora
  - peft
  - customer-support
---

# eval-gated Qwen3.8-27B, W4A16

## What it is

A QLoRA adapter for the support desk of a small mock shop, trained on 27B base weights,
merged into that base and requantized to W4A16 AWQ so it runs with vLLM on one A10.

The model writes a plan or a reply. It does not decide what reaches the customer. In the
project that trained it, a deterministic gate read the shop's own rows and approved,
refused or escalated each draft. This repository ships weights, not that gate. Read every
sentence this model produces as a draft no one has checked.

## What produced these weights

The adapter is run `full-20261114T201900Z` of the training project.

- Data: 263 written rows from 178 admitted teacher traces over a 284-case frozen
  benchmark for a mock shop.
- Training: 186 optimizer steps, rank 16, alpha 16, 496 modules, 116,727,808 trainable
  parameters.
- Loss: 0.2051 to 0.0840, first-five to last-five means over 17,868 supervised
  train-split tokens. That is a fit to those tokens, not a score of the model.

Merge and requantize: the LoRA delta was applied to the dequantized base in bfloat16,
then AWQ quantized to W4A16. What the merge checks showed:

- The adapter's parameters load bit for bit from `adapter_model.safetensors` (relative L2
  0.000e+00 against the file).
- 992 tensors changed, and every one of the 496 adapter modules is present in the merged
  checkpoint (0 absent).
- Nine modules sampled across depth carry their `B @ A` delta inside the bfloat16
  rounding ceiling. The other 487 modules were not sampled.

The served checkpoint is 17.31 GiB, with `model.safetensors` at 18,568,390,152 bytes.

Serving, measured on one A10: 4,096-token window, `max_num_seqs 8`,
`gpu_memory_utilization 0.92`, bfloat16, 19.03 of 22.06 GiB in use after load, 22.84
output tok/s on one composed prompt (316 prompt tokens, 30 output tokens). vLLM reported
1.08x concurrency at 4,096 tokens, which means two long requests cannot be in flight
together.

## Files

| Path | What it is |
| --- | --- |
| `model.safetensors` | W4A16 AWQ weights, 17.31 GiB, 18,568,390,152 bytes |
| `config.json` | model config, including the quantization config |
| `generation_config.json` | generation defaults |
| `recipe.yaml` | the AWQ quantization recipe |
| `processor_config.json` | processor settings |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | tokenizer and chat template |
| `adapter/adapter_model.safetensors` | the LoRA adapter, 467,062,560 bytes |
| `adapter/adapter_config.json` | rank 16, alpha 16, target modules |
| `adapter/tokenizer.json`, `adapter/tokenizer_config.json`, `adapter/chat_template.jinja` | the training run's tokenizer copy, not weights |
| `adapter/README.md` | the training run's own note |

## Run it

The quantized checkpoint, for vLLM:

```bash
vllm serve <account>/eval-gated-qwen3.8-27b \\
  --max-model-len 4096 \\
  --max-num-seqs 8 \\
  --gpu-memory-utilization 0.92 \\
  --dtype bfloat16
```

The adapter, on the base it was trained against. This is the adapter path, not the served
W4A16 path; the two are not bit-identical.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"  # a bitsandbytes 4-bit base
REPO = "<account>/eval-gated-qwen3.8-27b"

model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(model, REPO, subfolder="adapter")
tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="adapter")
```

## Measured and not measured

Measured, as the record has it:

- **Composed output shape: 14 of 14 cases.** Thirteen were in the training split and
  N-104 was the validation row, so this is a compatibility check on cases the corpus
  held, 13 of them seen. It shows the schema and the serving path preserved a trained
  behaviour. It is not a generalisation measure.
- **0 of 118 approved adversarial drafts from this model held a claim the log did not
  support**, against 2 of 36 from the base. A second model read the rows as adjudicator;
  no person read them. The honest control on the model's own normal-case drafts flagged
  4 of 115 (3.5 %), so the floor is not zero: one of those was ruled a false positive
  (N-102) and another was flagged unstable (N-81). Read the figure as "the adjudicator
  found no false claim", not as "there is none".

Not measured:

- Free-text quality on tickets outside the training corpus.
- Any accuracy or safety rating against a held-out human standard.
- Concurrency above one long request.
- Anything about a real shop. The shop, its orders and its customers are mock data.

## Intended use and limits

Research and demonstration, on a mock shop, behind a deterministic gate. Not a production
support agent, and not a source of truth about any order.

It can choose a wrong action and state it truthfully. On the demo's N-104 case it refunded
2,100 cents where 2,599 was required, and the reply matched the refund row it had just
written, so the gate approved a true sentence about a wrong decision. That is the design
the gate exists for. Without a gate, that reply reads like a correct one.

## Licence and attribution

Apache-2.0, inherited from the base model. The base weights are unsloth's release of
Qwen3.8-27B: `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`. The adapter, the merge, the
quantization and the measurements are this project's work.
'''

app = modal.App("eval-gated-lora-agent-export")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)

# huggingface_hub 2.4.0 pulls hf_xet, the supported chunked transfer path. The older
# `hf_transfer` extra is deprecated upstream for exactly this job.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]==2.4.0")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            # Container-local scratch for the Hub client. `/vol` stays read-only and
            # nothing is copied out of it but the bytes of the upload itself.
            "HF_HOME": "/root/.cache/hf",
        }
    )
)

export_kwargs = dict(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={VOLUME_MOUNT: volume.read_only()},
    cpu=8.0,
    memory=32768,
    timeout=7200,
    retries=0,
)


def _token() -> str:
    """The Hub token from the secret, or a loud failure that names no value."""
    for name in TOKEN_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    raise RuntimeError(
        f"no Hub token in the {SECRET_NAME} secret: expected one of {', '.join(TOKEN_VARS)} "
        "to be set. Add the key to the secret; this function prints key names only."
    )


def _whoami(token: str) -> dict:
    """Identity from `/api/whoami-v2`, which carries the account type and its orgs.

    `hf_api` 2.4.0 dropped `list_orgs`, and the identity fields the card needs
    (`type`, `orgs`, `plan`) are all on this one documented response.
    """
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "eval-gated-lora-agent-export"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _listing(path: str) -> dict:
    """Every file under `path`, relative to it, with its size in bytes."""
    found = {}
    if not os.path.isdir(path):
        return found
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            found[os.path.relpath(full, path).replace(os.sep, "/")] = os.path.getsize(full)
    return found


def _print_files(files: dict) -> None:
    for name in sorted(files):
        print(f"  {files[name]:>13,}  {name}")


@app.function(**export_kwargs)
def probe():
    """Who the token is, and what the Volume holds. Uploads nothing."""
    token = _token()
    who = _whoami(token)
    quant = _listing(QUANT_DIR)
    adapter = _listing(ADAPTER_DIR)

    return {
        "account": who.get("name"),
        "account_type": who.get("type"),
        "account_is_pro": bool(who.get("isPro")),
        "orgs": [org.get("name") for org in who.get("orgs", [])],
        "plan": who.get("plan"),
        # Names only. A value here is a credential.
        "env_keys_containing_token_or_hf": [
            key for key in sorted(os.environ) if "TOKEN" in key.upper() or "HF" in key.upper()
        ],
        "env_key_count": len(os.environ),
        "quant_dir": QUANT_DIR,
        "quant_dir_present": os.path.isdir(QUANT_DIR),
        "quant_files": quant,
        "quant_bytes": sum(quant.values()),
        "adapter_dir": ADAPTER_DIR,
        "adapter_dir_present": os.path.isdir(ADAPTER_DIR),
        "adapter_files": adapter,
        "adapter_bytes": sum(adapter.values()),
    }


@app.function(**export_kwargs)
def export(private: bool = True):
    """Create the repo private, upload both directories and the card, report what landed."""
    from huggingface_hub import HfApi, create_repo, upload_folder

    # Fail closed. The owner approved a private repo; a public one is out of scope,
    # so this refuses before anything exists rather than trusting the caller.
    if private is not True:
        raise PermissionError(
            "this export is private-only: the owner approved a private repo. "
            "Refusing to run with private=False."
        )

    token = _token()
    api = HfApi(token=token)
    account = _whoami(token).get("name")
    if not account:
        raise RuntimeError("the token carries no account name; refusing to guess a namespace")

    repo_id = f"{account}/{REPO_NAME}"

    for label, path in (("quantized artifact", QUANT_DIR), ("adapter", ADAPTER_DIR)):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} directory missing on the Volume: {path}")

    url = create_repo(repo_id=repo_id, token=token, private=True, exist_ok=True)

    # `exist_ok=True` would happily adopt a repo that already exists. If that repo is
    # public, stop here: nothing has been uploaded yet, so nothing leaks.
    info = api.model_info(repo_id=repo_id, token=token)
    if not info.private:
        raise PermissionError(
            f"{repo_id} exists and is public. This export will not add weights to a public "
            "repo. Make it private on the Hub, or change REPO_NAME, then run again."
        )

    quant = _listing(QUANT_DIR)
    adapter = _listing(ADAPTER_DIR)

    quant_commit = upload_folder(
        repo_id=repo_id,
        folder_path=QUANT_DIR,
        path_in_repo="",
        token=token,
        commit_message="serve: W4A16 AWQ artifact, 17.31 GiB",
        commit_description="The served checkpoint, straight from the Modal Volume.",
    )
    adapter_commit = upload_folder(
        repo_id=repo_id,
        folder_path=ADAPTER_DIR,
        path_in_repo="adapter",
        token=token,
        commit_message="adapter: full-20261114T201900Z",
        commit_description="The LoRA adapter the merge and the card describe.",
    )
    card_commit = api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
        commit_message="card: what this is, what produced it, what was measured",
    )

    remote = api.model_info(repo_id=repo_id, token=token, files_metadata=True)
    files = {s.rfilename: s.size for s in remote.siblings}

    return {
        "repo_id": repo_id,
        "url": str(url),
        "private": remote.private,
        "commits": {
            "quantized": getattr(quant_commit, "oid", None),
            "adapter": getattr(adapter_commit, "oid", None),
            "card": getattr(card_commit, "oid", None),
            "head": remote.sha,
        },
        "files": files,
        "total_bytes": sum(size for size in files.values() if size is not None),
        "source_bytes": {
            "quantized": sum(quant.values()),
            "adapter": sum(adapter.values()),
            "card": len(MODEL_CARD.encode("utf-8")),
        },
    }


@app.function(**export_kwargs)
def verify(repo_id: str):
    """Read the repo back with the same token and check it, rather than print it."""
    from huggingface_hub import HfApi

    token = _token()
    api = HfApi(token=token)
    info = api.model_info(repo_id=repo_id, token=token, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}

    checks = []
    if info.private is not True:
        raise PermissionError(f"{repo_id} is not private: private={info.private}")
    checks.append("private is True")

    listed = api.list_repo_files(repo_id=repo_id, token=token)
    if set(listed) != set(files):
        raise AssertionError("list_repo_files and model_info disagree on the file set")
    checks.append(f"list_repo_files agrees with model_info on {len(listed)} files")

    if files.get("model.safetensors") != QUANT_WEIGHTS_BYTES:
        raise AssertionError(
            f"model.safetensors is {files.get('model.safetensors')} bytes on the Hub, "
            f"expected {QUANT_WEIGHTS_BYTES}"
        )
    checks.append(f"model.safetensors is {QUANT_WEIGHTS_BYTES:,} bytes")

    if files.get("adapter/adapter_model.safetensors") != ADAPTER_WEIGHTS_BYTES:
        raise AssertionError(
            f"adapter/adapter_model.safetensors is "
            f"{files.get('adapter/adapter_model.safetensors')} bytes on the Hub, "
            f"expected {ADAPTER_WEIGHTS_BYTES}"
        )
    checks.append(f"adapter/adapter_model.safetensors is {ADAPTER_WEIGHTS_BYTES:,} bytes")

    expected = set(_listing(QUANT_DIR)) | {f"adapter/{name}" for name in _listing(ADAPTER_DIR)}
    expected |= {"README.md"}
    missing = sorted(expected - set(files))
    if missing:
        raise AssertionError(f"files the Volume holds but the repo does not: {missing}")
    checks.append(f"every one of the {len(expected)} source files is in the repo")

    # An anonymous read must be refused. 401 and 404 both satisfy that; a 200 does not.
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}",
        headers={"User-Agent": "eval-gated-lora-agent-verify"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            code = response.status
    except urllib.error.HTTPError as error:
        code = error.code
    if code == 200:
        raise AssertionError(f"an anonymous read of {repo_id} returned 200: the repo is readable")
    checks.append(f"anonymous read refused with HTTP {code}")

    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "private": info.private,
        "head": info.sha,
        "files": files,
        "total_bytes": sum(size for size in files.values() if size is not None),
        "checks": checks,
    }


@app.local_entrypoint()
def main(private: bool = True, probe_only: bool = False, verify_only: str = ""):
    if probe_only:
        result = probe.remote()
        print("probe")
        print(f"  account            {result['account']}")
        print(f"  account type       {result['account_type']}")
        print(f"  account is pro     {result['account_is_pro']}")
        print(f"  plan               {result['plan']}")
        print(f"  orgs               {result['orgs'] or 'none'}")
        print(f"  env keys (n={result['env_key_count']}), names only:")
        for key in result["env_keys_containing_token_or_hf"]:
            print(f"    {key}")
        print(f"  quantized dir      {result['quant_dir']} present={result['quant_dir_present']}")
        _print_files(result["quant_files"])
        print(f"  quantized bytes    {result['quant_bytes']:,}")
        print(f"  adapter dir        {result['adapter_dir']} present={result['adapter_dir_present']}")
        _print_files(result["adapter_files"])
        print(f"  adapter bytes      {result['adapter_bytes']:,}")
        return

    if verify_only:
        result = verify.remote(verify_only)
        print("verify")
        print(f"  repo               {result['repo_id']}")
        print(f"  url                {result['url']}")
        print(f"  private            {result['private']}")
        print(f"  head               {result['head']}")
        print(f"  files              {len(result['files'])}")
        _print_files(result["files"])
        print(f"  total bytes        {result['total_bytes']:,}")
        for check in result["checks"]:
            print(f"  ok                 {check}")
        return

    result = export.remote(private)
    print("export")
    print(f"  repo               {result['repo_id']}")
    print(f"  url                {result['url']}")
    print(f"  private            {result['private']}")
    for name, oid in result["commits"].items():
        print(f"  commit {name:<10} {oid}")
    print(f"  files              {len(result['files'])}")
    _print_files(result["files"])
    print(f"  total bytes        {result['total_bytes']:,}")
    print(f"  source bytes       {result['source_bytes']}")
