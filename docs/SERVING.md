# SERVING — stage 6 slice 1: is the locked A10 target reachable?

Status: slice 1 measured 2026-11-15; the endpoint and the demo deployed 2026-11-16
(`## Deployed endpoint and demo`, below the measurement record). No commit.
Run order, code and cost bounds: `serve/feasibility.py`, `serve/app.py`, `serve/demo.py`.
Raw logs: `out/serve-*.log`.

`SPEC.md` §2 locks serving to **Modal vLLM on an A10 24 GB, scale to zero**. The
trained artifact (`runs/full-20261114T201900Z/adapter`) is a bitsandbytes 4-bit
QLoRA adapter, which vLLM cannot read. This slice measured the only path from
there to vLLM: dequantize and merge to 16-bit, requantize to a vLLM format, then
load on the A10.

## The budget, from the files (scan, $0.0001)

`modal run serve/feasibility.py::scan_cli` reads the base checkpoint's
safetensors headers — shapes only, no weights downloaded.

| Family | Parameters | Stored | Tensors |
|---|---|---|---|
| MLP | 17,112,760,320 | packed 4-bit | 192 |
| linear attention | 5,562,046,464 | packed 4-bit | 336 |
| full attention | 1,677,729,792 | packed 4-bit | 96 |
| `lm_head` | 1,271,398,400 | bf16 | 1 |
| `embed_tokens` | 1,271,398,400 | bf16 | 1 |
| vision tower | 460,384,128 | bf16 | 167 |
| other | 660,480 | bf16 | 129 |
| **total** | **27,356,377,984** | | |

Consequences that decide the slice before any GPU is booked:

- The base artifact on disk is **22.34 GB**. That alone leaves 1.7 GB of an A10
  for KV cache, CUDA graphs and workspace: **the shipped artifact cannot be
  served on the locked GPU at any sequence length.**
- A W4A16 requantize keeping `lm_head`, `embed_tokens`, vision and MTP in bf16
  projects to **17.29 GiB** — 11.69 GiB quantized at **4.125 bits per
  parameter** (4 for the values plus one 16-bit group scale per 128 values,
  16/128 = 0.125) plus 5.59 GiB retained. The artifact that was actually built is
  17.312 GiB, so the arithmetic lands within 0.02 GiB.
- If linear attention were kept in bf16 as well, the projection is **24.74 GiB**,
  over 24 GB before any cache. Linear attention must be quantized for the A10 to
  have a chance, and that is the route the artifact took.
- KV is **64 KiB/token** (16 full-attention layers of 64, 4 KV heads, head_dim
  256). That figure stands, but it is only half the story on a hybrid model: the
  linear-attention half keeps a fixed **recurrent state of 144.0 MiB per
  sequence in float32** (`mamba_ssm_dtype`) plus a **5.62 MiB convolution
  state**, so **149.6 MiB per sequence** before a single token is cached. Batch
  8 costs 1.17 GiB of state; the default batch of 256 costs **37.4 GiB**, which
  is why the first fit rung died (below).
- At 0.92 of 24 GB, 17.29 GiB of weights leave **4.79 GiB**. The naive KV
  arithmetic says 78,503 tokens; vLLM's own allocator, which also pays the state,
  reports **4,437**. Take the measured figure.

## Merge — CPU, 8 cores / 112 GiB. Bound $0.5302, measured **$0.0991**

`modal run serve/feasibility.py::merge_cli`. Worst case printed before the run:
**$0.5302** at $1.2724/hr (cpu $0.1572 + memory $0.3730), timeout 1500 s,
`retries=0`. CPU is the cheapest container that holds the work: the peak is the
20.8 GiB packed base plus the 51.0 GiB bf16 result. A 40 GB A100 cannot hold it
and an 80 GB A100 costs $2.4984/hr against $1.2724/hr.

| Step | Measurement |
|---|---|
| One-layer proof (1024×5120) | 5,242,880 values dequantized, **7.98-19.31 s per billion values**; predicted 218-528 s for the whole model |
| Load 4-bit base, CPU | **6.4 s** (mmap), 32.0 GiB resident |
| Dequantize | **352** 4-bit linears → bf16 in **98.9 s**, peak **72.9 GiB** of the 112 GiB request |
| Adapter on disk | 992 LoRA tensors, **116,727,808** parameters, 467,062,560 bytes |
| Adapter loaded | relative L2 **0.000e+00** against `adapter_model.safetensors` — the run's own weights, bit for bit |
| Merge | **496** modules in **54.7 s** |
| Merge proof, one module (that run) | delta moved 2.441e-04, norm ratio 0.973 of `B @ A`, cosine 0.795, max element error 1.41e-04 against the bf16 ceiling 6.56e-03 |
| Save | 1,184 tensors, **50.97 GiB in 14 files**, 59.5 s |
| **Wall** | **280 s**, cost **$0.0991** |

What the log actually shows at the end, because it matters: the run printed
`merge: wall 280s, peak 72.9 GiB` and then **`KeyError: 'key'`**. That is the
function's return dictionary — it referenced a proof field renamed during the
work — and it fired *after* `volume.commit()`. The merge completed and the
checkpoint on the Volume is whole (1,184 tensors, 50.97 GiB, 14 files), but the
run's JSON was never returned. A later run fixed the return; the stored artifact
is the one this slice verified, not that later code path.

The merge run's own proof is **one module**, not five: that is all the log for
that run contains. The nine-module check is separate evidence gathered
afterwards against the stored checkpoint, and it must not be read back onto that
run. What the original run proved is one module merged correctly plus a load that
was bit-exact; what the later verification proves is nine modules across depth
plus key accounting over the whole file. Neither proves the other.

| Re-check of the stored checkpoint (24 s, $0.006) | Value |
|---|---|
| Tensors found | 1,184 across 7 shards |
| Modules sampled across depth | **9** — layers 3, 11, 19, 27, 35, 43, 51, 59, 63 |
| Delta norm ratio, per module | 0.964 to 1.019 of `B @ A` |
| Cosine, per module | 0.776 to 0.862 |
| Max element error, per module | 1.22e-04 to 1.82e-04, every one under its bf16 ceiling (6.6e-03 to 1.2e-02) |
| Adapter modules absent from the checkpoint | **0 of 496** |

What this proves and what it does not: nine weights across the depth received
their `B @ A` delta, and every LoRA tensor in the file belongs to a module the
checkpoint holds. **It does not prove that the 487 unsampled adapted modules each
received theirs** — that would take a per-module sweep of all 496, which this
slice did not run.

Three things the merge cost, all found by the run and worth knowing before
anyone repeats it:

1. **`AutoModelForCausalLM` silently drops this adapter.** It loads the text
   tower as `model.layers.*`; the adapter is keyed `model.language_model.layers.*`.
   PEFT matches no key, keeps its fresh initialisation, and still reports
   116,727,808 attached parameters. The proof caught it (relative L2 1.406, the
   signature of uncorrelated tensors). Loading
   `Qwen3_5ForConditionalGeneration` — the architecture in the checkpoint's own
   config — makes the load bit-exact.
2. **`merge_and_unload()` cannot be used here.** On a bitsandbytes checkpoint it
   would add a delta into packed uint8 weights, and on this model it raises
   `KeyError(slice(None, None, -1))`. The merge is written out explicitly:
   `base_layer.weight += (B @ A) * alpha / r`.
3. **`save_pretrained` cannot be used here.** Transformers tries to reverse the
   checkpoint conversion it applied on load and raises `NotImplementedError`
   inside `revert_weight_conversion`. The state dict is written directly, with
   `.base_layer` and the LoRA factors removed and `quantization_config` set to
   null, so nothing downstream tries to add the adapter twice.

## Quantize — A100 80 GB. Bound $1.8215, measured **$0.432**

`modal run serve/feasibility.py::quant_cli`. Worst case printed before the run:
**$1.8215** at $3.6429/hr (gpu $1.2492 + cpu $0.1886 + memory $0.3836), timeout
1800 s, `retries=0`. 80 GB is the cheapest GPU that holds a 51 GiB bf16 model
plus the quantizer's copies; llm-compressor 0.15.0, `AWQModifier` scheme W4A16,
group size 128.

| Measurement | Value |
|---|---|
| Model on the GPU | 50.96 GiB, `Qwen3_5ForConditionalGeneration`, `Qwen3VLProcessor` |
| Calibration | 32 samples of up to 1,024 tokens from the 263 SFT records |
| Ignore list | `lm_head`, `.*embed_tokens.*`, `.*visual\..*`, `.*\.mtp\..*`, `.*conv1d.*` |
| Modules quantized | 496 of 496 compressed |
| MTP | **copied unchanged** — llm-compressor warned "MTP layers are not loaded for quantization and will be copied unchanged when saving" |
| Quantize | **399 s**; wall **427 s** |
| Artifact | **17.31 GiB** in 8 files at `/vol/serve/slice1/awq-w4a16` |
| Cost | **$0.432** |

Provenance, because the local log for this step looked truncated: the pipe
buffered the calibration bars, but Modal keeps the run's own stdout. The
evidence is two commands, not two files:

- `modal app logs ap-JrfLtEUT7TSek4ADEeRDbW` printed `quantize: done in 399s`,
  `artifact: 17.31 GiB in 8 files` and
  `quantize: wall 427s, {"seconds": 426.9, … "total_usd": 0.432}`.
- `modal volume ls eval-gated-lora-agent /serve/slice1/awq-w4a16 --json` returned
  eight entries totalling 18,588,402,779 bytes — **17.312 GiB** — with
  `model.safetensors` alone at 18,568,390,152 bytes.

Two independent sources, one number. Re-running the two commands reproduces both.

17.312 GiB against the 17.29 GiB projection at 4.125 bits: the artifact is
0.02 GiB over budget, and the old 4.5-bit projection that preceded the
correction overshot it by a GiB. The A10 keeps 4.79 GiB of the 22.08 usable.

## Fit on the A10 — bound $0.5153, measured **$0.2159**

`modal run serve/feasibility.py::fit_cli`. Worst case printed before the run:
**$0.5153** at $1.5460/hr (gpu $0.3672 + cpu $0.0629 + memory $0.0852), timeout
1200 s, `retries=0`. The probe runs as a **fresh subprocess**, so "cold start" is
an interpreter that has never held this model. It excludes container start, image
pull and scale-from-zero scheduling — `SPEC.md` §13.3 stays open.

**It loads at batch width 8. 256 failed; 8 succeeded; the widths between were
not tested.** Two rungs were measured:

| Rung | Result |
|---|---|
| `max_model_len 4096`, default `max_num_seqs` (256) | **CUDA OOM**: tried to allocate 12.25 GiB with 3.74 GiB free and 18.31 GiB already held |
| `max_model_len 4096`, `max_num_seqs 8` | **loaded** in **189.3 s** |

| Measurement | Value |
|---|---|
| GPU | NVIDIA A10, 22.06 GiB usable of 24 GB nameplate |
| Weights on GPU | 18.31 GiB held during the failed rung's allocation |
| VRAM after load | **19.03 GiB used, 3.03 GiB free** |
| KV cache, from vLLM itself | **4,437 tokens**, "maximum concurrency for 4,096 tokens per request: **1.08x**" |
| Cold start, engine init | **189.3 s** (the probe's 485.7 s spans the failed rung first) |
| Speed, N-105 composed prompt | 316 prompt tokens, 30 output tokens, 1.31 s → **22.84 output tok/s**, 263 tok/s including prefill |
| Cost | **$0.2159**, wall 503 s |

Two honesty notes. `torch_max_memory_allocated_gib` reads 0.0 because vLLM runs
the model in its own EngineCore process; the VRAM figure is `cudaMemGetInfo`
across the device, which is the number that matters. And the first OOM is not a
weight problem: 18.31 GiB of weights fit, and the rejected allocation is the
recurrent-state and block table for 256 sequences — at the measured 149.6 MiB per
sequence, 256 sequences need **37.4 GiB** of state alone.

Reaching the locked target therefore takes one configuration change, not an
infrastructure change: `max_num_seqs 8`. What it buys and what it costs:

- 8 is the only width that was tested and loaded. 256 is the only width that was
  tested and failed. Whether 16, 32 or 64 would load is **untested** — the
  arithmetic suggests somewhere near 20 sequences of state fit in the 3.03 GiB
  left after load, before KV competes for it;
- the gate's longest prompt is 2,757 tokens, so **one long request at a time** is
  the operating point; vLLM reports 1.08x concurrency at 4,096 tokens, so two
  long requests cannot be in flight together;
- 22.84 output tok/s on the composed path is a demo, not a desk;
- 189 s engine init plus container start is the cold-start bill that
  scale-to-zero pays on every wake.

Route that failed first, for the record: vLLM 0.33.0 JIT-compiles a kernel for
the `qwen3_5` hybrid path at engine init and refuses to start without `nvcc`
(`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`).
`flashinfer-cubin`, `ninja` and `build-essential` did **not** fix it; a
`nvidia/cuda:13.0.1-devel` base image did. That is an image requirement the
deploy has to carry.

## Quality: the composed suite

**14/14 parse ok, 14/14 match `required_calls`** on the quantized model — the same
score the full-precision adapter records in `out/modal-check5.log`. All fourteen
cases also reproduce the trained target exactly (`equals_target: True`).

Read it for what it is: **a compatibility check, not a quality claim.** Thirteen
of the fourteen cases were in the training split and N-104 was the validation
row, so every case is seen; the quantized artifact reproduces the adapter's
outputs on seen data, which shows the merge, the requantize and the serving path
preserved behaviour. It does not measure generalisation, and it says nothing
about the held-out set `SPEC.md` §7 still leaves unmeasured.

## Cost

List rates read from modal.com/pricing on 2026-11-15: A10 **$1.1016/hr**,
A100 40 GB $2.0988/hr, A100 80 GB $2.4984/hr, CPU $0.0000131/core/s, memory
$0.00000222/GiB/s. Every figure below is wall time × list rate from the run's own
printed JSON — an estimate, never a bill. `retries=0` on every function.

| Step | Container | Bound printed first | Measured | Source |
|---|---|---|---|---|
| scan (headers only) | CPU 2 cores / 8 GiB | $0.0264 | **$0.0000** (0.6 s) | printed JSON |
| merge | CPU 8 cores / 112 GiB | $0.5302 | **$0.0991** (280 s) | printed JSON |
| quantize | A100 80 GB | $1.8215 | **$0.4320** (427 s) | printed JSON, app log |
| fit + suite | A10 | $0.5153 | **$0.2159** (503 s) | printed JSON |
| **four measurements** | | **$2.8934** | **$0.7470** | arithmetic over the four |
| verify_merge (re-check) | CPU 8 cores / 64 GiB | $0.2222 | **$0.0060** (24 s) | printed JSON |

The four-measurement total is the sum of the four rows above and nothing else.
The re-check is separate because it was run during review, not during the
original slice.

**The rest of the slice's spend is unaudited.** The path did not work first time
— seven merge attempts, a quantize run that reached the save and lost its client,
two fit runs killed by the missing compiler — and none of those attempts had a
cost record of their own. There is no per-app ledger in this repo to add up, so
this document reports no figure for them. What is supported is $0.7470 of
measurement and $0.0060 of re-check; the learning cost is real, is not measured,
and is not estimated here either. `SPEC.md`'s $1-2 target for the slice cannot be
checked against evidence as it stands.

Storage: 50.97 GiB (merged bf16) + 17.31 GiB (AWQ) on the Volume. Volumes are
$0.09/GiB/month with 1 TiB included, so this is inside the free allowance. The
merged bf16 checkpoint is the reusable half: a second quantizer would not need to
re-merge.

## Deployed endpoint and demo (2026-11-16)

Both apps are live on Modal in `us-east`, deployed from `serve/app.py` and
`serve/demo.py`. The locked target is unchanged: one A10, `max_model_len 4096`,
`max_num_seqs 8`, `gpu_memory_utilization 0.92`, `min_containers=0`,
`max_containers=1`, `scaledown_window=60`. Phase B is off in the payload because
the DeepSeek judge sits behind the subscription key, and that key does not go into
a remote sandbox; the deployed agent answers with Phase A and the gate. Modal
refuses `retries` on a Web Function (`InvalidError: Web Functions do not support
retries`), so the HTTP door is single-shot by construction. The code also runs
one `evaluate` per request and never wraps a mutating action in a retry loop — a
retried refund would refund twice.

| App | Door |
|---|---|
| `eval-gated-lora-agent` | `POST https://daniel-osman--eval-gated-lora-agent-agent-run.us-east.modal.run` — the model |
| | `GET https://daniel-osman--eval-gated-lora-agent-health.us-east.modal.run` — presets, model-free |
| `eval-gated-lora-agent-demo` | `https://daniel-osman--eval-gated-lora-agent-demo-web.us-east.modal.run` — page, `/health`, `/run` |

The demo app is CPU-only (`fastapi` + `httpx`). It serves `serve/demo.html`,
proxies the agent's health endpoint, and calls `Agent.serve` over the Modal SDK —
one hop, no HTTP redirect (defect 1 below).

### Measured on this deployment

| What | Call | Result | Saved |
|---|---|---|---|
| Cold start, demo path, no agent container | `modal container list --json` → **0** agent containers, then `POST /run {"scenario":"N-100"}` | **309 s wall**; `engine.init_seconds 291.4`; `engine.request_number 1`; `timings.total_seconds 0.94`; verdict `APPROVED`; `matches_required_calls true` | `out/serve-deployed3-cold.json` |
| Cold start, HTTP door (earlier) | `POST /agent-run {"scenario":"N-100"}` | **312.73 s** | `out/cold.json` |
| Warm, back to back, demo `/run` | same door: N-101, N-103, N-105 | **3 s** (0.931 s in request) · **2 s** (1.32 s) · **3 s** (1.32 s) | — |
| Warm free text, demo `/run` | `POST /run {"ticket":"Hi, my order ORD-1013 has not arrived. What is going on?"}` | **9 s wall**, **4.432 s** in request, **2 generations**; plan `[{"name":"get_order","args":{"order_id":"ORD-1013"}}]`; reply "Your order ORD-1013 for one Linen throw ($34.99) shows as paid, and its DHL shipment SHP-1013 is marked delivered with tracking number TRK1013." | `out/serve-demo-freetext.json` |
| Scale-down | `modal container list --json` after ~2 min idle | only the demo app's container; **agent containers: 0** | — |
| Unknown preset, through `/run` | `POST /run {"scenario":"N-999"}` | **HTTP 400**, detail quoted below | — |
| Browser, `Run N-104` | click on the demo page | panel filled: plan, call, gate verdict, reply, timings, versions, UNPROVEN and Phase B blocks | `out/demo-n104.png` |

Warm latency is the request's own `timings.total_seconds` against wall time on
the same call; the wall figure includes the demo hop.

### Post-deploy first request, then warm

`modal deploy serve/app.py` finished at 20:1x on 2026-11-16. The first request
after it, through the demo's `/run`:

| What | Call | Result |
|---|---|---|
| Post-deploy first request, unseen preset | `POST /run {"scenario":"N-110"}` | **384 s wall**, **HTTP 200**; verdict **REJECTED**; `matches_required_calls false`; `training_seen false`; `unproven` = "this preset is a frozen bench case the adapter never trained on; a match is a serving-compatibility result, not generalisation" |
| Then warm, seen preset | `POST /run {"scenario":"N-100"}` | **HTTP 200** in **3 s**; **APPROVED**; `matches_required_calls true`; `training_seen true`; `unproven[0]` = "presets are frozen training-seen bench scenarios; reproducing them is a compatibility check, not a quality claim" |

Two cold figures, and they measure different things: **309 s** is the idle
scale-from-zero measurement (0 agent containers, 291.4 s of engine init), and
**384 s** is the post-deploy first request. The difference is image pull and
container start: the deploy was fresh, so its image was not on the host yet.

What that shows, in one sentence: the six unseen presets are labelled as unseen in
the response, and at least one of them (N-110) is rejected by the gate — the demo
does not only show passes.

### Scale-down, the sequence to quote

`modal container list --json` printed **0** agent containers before the cold
call, **1** GPU container while it ran, and **0** again after roughly two minutes
idle. Scale-to-zero is therefore real on this deployment, and it is what the next
caller pays 291 s of engine init for.

### The 400, in the caller's words

```
{"detail":"unknown scenario; the composed presets are: N-100, N-101, N-102, N-103, N-104, N-105, N-106, N-107, N-108, N-109, N-110, N-81, N-82, N-83, N-86, N-88, N-89, N-97, N-98, N-99"}
```

Status **400**. The text is the endpoint's own wording, carried through the demo
unchanged: `serve` returns a status and a body as JSON text, and `/run` mirrors
them. That the status survives at all is the visible half of the RPC door; the
invisible half is the cold POST that no longer dies in a redirect.

### The browser

`Run N-104` — the validation row, the one composed case not in the training
split — fills the whole panel:

| Panel part | Value |
|---|---|
| Verdict | `APPROVED` |
| Match | `matches_required_calls: false — plan mismatch: refund amount 2100 != 2599` |
| Boundary block | "WRONG DECISION — APPROVED BY THE GATE", with the one-sentence explanation |
| Plan | the composed plan the model wrote |
| Call | `refund_order`, `REF-1014`, `remaining_cents 499` |
| Gate | the verdict JSON |
| Reply | "We refunded $21.00 for ORD-1023." |
| Timings | 1.31 s in request; engine init 290.5 s; request #3 |
| Stack | vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2 |
| Honesty | the UNPROVEN line and the Phase B off block |

This is the demo's best frame and its most honest one: a refund of 2100 cents
where 2599 was required, approved because Phase A's rules found nothing to
contradict, and drawn as a wrong decision rather than hidden. The page shows what
the gate did. It does not show the model being right.

### Two defects, both fixed, both the interesting part

1. **A 303 on the wire is not escapable.** A cold or queued POST to the web
   function answers `303` to the same URL plus a `__modal_attempt_token`. A
   normal redirect follower turns 303 into GET and drops the body; a re-POST and
   a GET were both refused with `400 modal-http: bad redirect method`. The hop
   was removed rather than chased: the demo calls `Agent.serve` through the SDK,
   and the SDK queues without a redirect. `serve/demo.py` carries no
   `post_following_303`, no `MAX_REDIRECT_HOPS`, no `AGENT_URL`.
2. **A dict return value cannot cross into a torch-free image.** The first
   `Agent.serve` returned a dict. The demo image holds no torch, so unpickling
   the result failed in the **success** path: `modal.exception.DeserializationError:
   Deserialization failed because the 'torch' module is not available in the
   local environment` (raised from `_utils/function_utils.py:577`). The fix
   returns JSON text from `serve` and parses it in the demo, and a test asserts a
   non-JSON value raises out of `serve` instead of being coerced.

Both defects lived between two containers, not inside either: the fresh-context
review of `serve/` would not have caught either one.

### The flaw still open

**Validation runs inside the model container.** An unknown preset id wakes the
A10, pays the ~5-minute cold start, and only then answers 400. The page offers
only known presets, so a visitor cannot reach it; a caller can. The cheap fix is
to validate `scenario` against the preset list in the demo app, which already
proxies `/health`, or in a small CPU function. It is named here as the next
improvement and was not built.

### Spend, metered

`python -m modal billing summary --for "this month" --json` printed metered
**17.18067970**, billed **0.00** (credits). Baselines: `out/serve-billing-before.json`
**14.36375475**, `out/serve-billing-after.json` **14.78476210**. The first
endpoint measurement cost **$0.42100735**; the whole deploy and demo verification
since then is **17.18067970 − 14.78476210 = $2.39591760**. That figure includes
image builds, failed attempts and four cold starts. It is metered, not estimated,
and not itemised.

## The decision the numbers force

The option taken on 2026-11-16 is the first one: the A10 stays, and the deployed
app is a single-user demo at `max_num_seqs 8`. The section above is that
decision, running.

**The locked A10 target is reachable — at one request at a time — and that is a
product decision, not an engineering one.**

The chain is measured end to end: 4-bit bnb adapter → 51.0 GiB bf16 merge →
17.31 GiB AWQ → loaded on an A10 at `max_num_seqs 8`, 19.03 GiB of 22.06, 4,437
KV tokens, 22.84 output tok/s, 189 s cold start, 14/14 on the composed suite.

Two ways to read it, and Daniel owns the choice:

1. **Keep the A10 and ship a single-user demo.** Set `max_num_seqs 8` in the
   serving app, carry a CUDA devel base image, and accept 22.8 tok/s and one
   concurrent long request. No infra change, no new rate, the $30/month credit
   covers it. The caveat is measured, not assumed: 256 sequences failed, 8
   loaded, and nothing between the two was tried.
2. **Move to a 48 GB or 80 GB GPU** for concurrency worth the name: L40S at
   $1.9512/hr or A100 40 GB at $2.0988/hr, roughly 1.8-1.9x the A10 rate, with
   room for the state that 256 sequences need — 37.4 GiB of it.

The one thing the numbers forbid is leaving this open: `SPEC.md` §2 says A10 with
no qualification, and an A10 at `max_num_seqs 8` serves one 4k request at a time,
which is not what "serve" usually means. Either the spec says "single-user demo,
`max_num_seqs 8`", or the serving line changes GPU. That is an infrastructure
change and it is Daniel's call, so slice 1 named it and stopped; the deployed
section above is the first option, running.
