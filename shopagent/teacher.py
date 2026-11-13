"""Teacher client for the OpenCode Go subscription API.

One job: take a system prompt and a user prompt, return the model's text. The
rest of the pipeline owns prompt building and admission; this module owns the
wire and nothing else.

Measured on this box (SPEC.md §2):

* gateway ``https://opencode.ai/zen/go/v1``, chat path ``/chat/completions``
* ``Authorization: Bearer $OPENCODE_API_KEY``; ``OPENCODE_GO_API_KEY`` is a
  fallback only. Neither name is ever written to a file or a log line.
* an ``x-opencode-session`` header is required; one stable id per run
* 8 concurrent requests succeeded, so concurrency is capped at 8
* 429 and 5xx are retried with backoff; any other 4xx at most once

The key is read from the environment at construction and held only in memory.
``repr`` redacts it, and no error message includes a header.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .tools import TOOL_SPECS

GATEWAY = "https://opencode.ai/zen/go/v1"
CHAT_PATH = "/chat/completions"
MODEL = "deepseek-v4.1-flash"
# Revision of the per-case brief text. The template digest below covers the
# system prompt and the stage instructions; it cannot see a change to
# `build_scenario_brief` or `build_reply_brief`, so the manifest records this
# number beside it. Bump it whenever a brief's wording changes what the teacher
# is told.
BRIEF_REVISION = 2
MAX_CONCURRENCY = 8
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (1.0, 2.0)

# A 403 appeared once on a first attempt and cleared on the next, so 403 gets a
# single retry. These statuses mean "wrong account, wrong path or wrong body";
# repeating them only burns time.
NO_RETRY_STATUSES = frozenset({400, 401, 402, 404, 405, 422})

# The gate owns the commit; a plan that calls send_reply would write its own
# text and bypass the reply contract.
ALLOWED_PLAN_TOOLS = tuple(spec["name"] for spec in TOOL_SPECS if spec["name"] != "send_reply")
_TOOL_CATALOGUE = json.dumps(
    [spec for spec in TOOL_SPECS if spec["name"] in ALLOWED_PLAN_TOOLS],
    indent=2,
    sort_keys=True,
)

MUTATING_TOOLS = ("refund_order", "create_reshipment", "update_address")
ACTION_BY_TOOL = {
    "refund_order": "refund",
    "create_reshipment": "reshipment",
    "update_address": "address",
}

SYSTEM_PROMPT = f"""You are the support model for a small online shop. You see
one customer ticket at a time and you answer it in two steps.

Step 1. Write the customer message that states the situation, and the tool plan
that resolves it. Step 2. Read the tool results and write the reply the
customer will read.

Available tools. Call only these, with exactly these argument names:
{_TOOL_CATALOGUE}

Rules:
- Never call send_reply. A separate gate sends the reply after it checks it.
- Read the order before a refund, a reshipment or an address change.
- Use only the ids and amounts named in the situation. Never invent an order,
  an amount, a refund, a shipment, a tracking number or an address.
- The reply states only what the tool results show.
- A failed action is not a success. Say so plainly.
"""

STAGE1_INSTRUCTIONS = """Write the customer message for this scenario in your
own words, and the tool plan that answers it. Return one JSON object and
nothing else: {"ticket": "...", "plan": ...}

The plan is a list of calls: [{"name": "get_order", "args": {"order_id": "ORD-1001"}}]"""

STAGE2_INSTRUCTIONS = """Write the customer reply. State only what the tool
results show: the order id, the item, the amount, the refund or shipment row,
the tracking number, the status. One short paragraph. Return one JSON object and
nothing else: {"reply": "..."}"""


def status_order_id(case: dict) -> str | None:
    """The order a status case is about.

    A `get_order` call names it. Four composed status cases carry no calls at
    all, so the order comes from `state_expect` instead. The brief and the
    admission match read this one function, so they cannot drift.
    """
    for call in case.get("required_calls", []):
        if call.get("name") == "get_order":
            order_id = call.get("args", {}).get("order_id")
            if order_id:
                return order_id
    for expectation in case.get("state_expect", []):
        if expectation.get("table") == "orders":
            order_id = expectation.get("match", {}).get("id")
            if order_id:
                return order_id
    return None


def expected_action(case: dict) -> str:
    """Which composed action the case needs, from its mutating call."""
    for call in case.get("required_calls", []):
        if call.get("name") in ACTION_BY_TOOL:
            return ACTION_BY_TOOL[call["name"]]
    return "status"


def _money(cents: int) -> str:
    return f"${int(cents) // 100}.{int(cents) % 100:02d}"


def _wants(case: dict) -> str:
    action = expected_action(case)
    return {
        "refund": "a refund",
        "reshipment": "a replacement shipment",
        "address": "a change to the delivery address",
        "status": "the status of an order",
    }[action]


def build_scenario_brief(case: dict, *, path: str) -> str:
    """The step-1 brief. It never contains the frozen ticket text."""
    lines = [
        f"SCENARIO {case['id']} ({case['kind']})",
        f"Title: {case.get('title', '')}",
        f"Customer needs: {_wants(case)}",
    ]
    order_id = status_order_id(case)
    if order_id:
        lines.append(f"Order: {order_id}")
    for call in case.get("required_calls", []):
        if call.get("name") == "refund_order" and "amount_cents" in call.get("args", {}):
            lines.append(f"Amount: {_money(call['args']['amount_cents'])}")
        if call.get("name") == "update_address":
            args = call.get("args", {})
            if args.get("address_id"):
                lines.append(f"Address row: {args['address_id']}")
            # Composed address names an existing row and carries no text; showing
            # the street there invites `line1`, which the plan refuses.
            if path == "free_text" and args.get("line1"):
                lines.append(f"New street: {args['line1']}, {args.get('city', '')}".strip(", "))

    if path == "free_text":
        calls = case.get("required_calls", [])
        if calls:
            lines.append("")
            lines.append("The scenario needs exactly these tool calls, in this order:")
            for call in calls:
                lines.append(
                    f"- {call['name']}({json.dumps(call.get('args', {}), sort_keys=True)})"
                )
            lines.append("Call each one once, in this order. Add no call and drop none.")

    faults = case.get("faults", [])
    if faults:
        lines.append("Faults in this session (the tool will answer this way):")
        for fault in faults:
            lines.append(
                f"- {fault['operation']} on {fault['key']} fails with {fault['mode']}"
                f" ({fault.get('times', 1)} time(s))"
            )
    else:
        lines.append("Faults in this session: none")

    if path == "free_text":
        facts = case.get("required_facts", [])
        if facts:
            lines.append("Your reply must state these values, word for word:")
            lines.extend(f"- {fact}" for fact in facts)

    if path == "composed":
        lines.append(
            'Return the plan as one object, not a list: {"action": "refund", '
            '"order_id": "ORD-1001", "amount_cents": 2400}. Actions: status, '
            "refund, reshipment, address. Ids and amounts only, never text."
        )
    lines.append("")
    lines.append(STAGE1_INSTRUCTIONS)
    return "\n".join(lines)


def build_reply_brief(*, ticket: str, plan, results: list[dict], required_facts=()) -> str:
    """The step-2 brief: the generated ticket, the plan and the real envelopes.

    The required values repeat here because this is the brief the reply is
    written from. Listing them only in the scenario brief let the teacher write
    a true sentence in its own words and miss the value the case needs.
    """
    parts = [
        f"CUSTOMER MESSAGE:\n{ticket}",
        f"PLAN:\n{json.dumps(plan, indent=2, sort_keys=True)}",
        f"TOOL RESULTS:\n{json.dumps(results, indent=2, sort_keys=True)}",
    ]
    facts = list(required_facts)
    if facts:
        parts.append("Your reply must state these values, word for word:")
        parts.extend(f"- {fact}" for fact in facts)
    parts.append("")
    parts.append(STAGE2_INSTRUCTIONS)
    return "\n".join(parts)


def _load_object(text: str, keys: set[str]) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TeacherParseError(f"teacher output is not JSON: {exc}", text) from exc
    if not isinstance(payload, dict):
        raise TeacherParseError("teacher output must be one JSON object", text)
    if set(payload) != keys:
        raise TeacherParseError(
            f"teacher output keys must be {sorted(keys)}, got {sorted(payload)}", text
        )
    return payload


def parse_stage1(text: str) -> tuple[str, object]:
    """`{"ticket": str, "plan": list|object}` -> (ticket, plan)."""
    payload = _load_object(text, {"ticket", "plan"})
    ticket = payload["ticket"]
    if not isinstance(ticket, str) or not ticket.strip():
        raise TeacherParseError("teacher ticket is empty", text)
    plan = payload["plan"]
    if not isinstance(plan, (list, dict)):
        raise TeacherParseError("teacher plan must be a list of calls or one object", text)
    return ticket.strip(), plan


def parse_stage2(text: str) -> str:
    """`{"reply": str}` -> reply."""
    payload = _load_object(text, {"reply"})
    reply = payload["reply"]
    if not isinstance(reply, str) or not reply.strip():
        raise TeacherParseError("teacher reply is empty", text)
    return reply.strip()



class TeacherUnavailable(RuntimeError):
    """The teacher cannot be reached, is not configured, or refused the call."""


class TeacherParseError(ValueError):
    """The response was reachable but not a usable chat completion."""

    def __init__(self, message: str, text: str = "") -> None:
        super().__init__(message)
        self.text = text


@dataclass(frozen=True)
class TeacherReply:
    text: str
    model: str
    attempts: int
    latency_ms: int
    status: int


def api_key_from_env() -> str:
    """The key, primary name first. Reading does not expose it."""
    key = os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_GO_API_KEY")
    if not key:
        raise TeacherUnavailable(
            "no teacher key: set OPENCODE_API_KEY (OPENCODE_GO_API_KEY is accepted "
            "as an alias). The key is never read from a file."
        )
    return key


def template_digest() -> str:
    """Stable id for the prompt template, recorded in the corpus manifest."""
    joined = "\n--stage1--\n".join([SYSTEM_PROMPT, STAGE1_INSTRUCTIONS, STAGE2_INSTRUCTIONS])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _default_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    """POST and return (status, body). HTTP errors are values, not exceptions."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _extract_text(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TeacherParseError(f"teacher response is not JSON: {exc}", text) from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TeacherParseError("teacher response has no message content", text) from exc
    if not isinstance(content, str) or not content.strip():
        raise TeacherParseError("teacher message content is empty", text)
    return content


class TeacherClient:
    """One chat call per prompt, with a bounded number of retries."""

    def __init__(
        self,
        *,
        session_id: str,
        model: str = MODEL,
        base_url: str = GATEWAY,
        api_key: str | None = None,
        transport=None,
        sleep=None,
        timeout: float = 60.0,
        max_attempts: int = MAX_ATTEMPTS,
        max_concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be one non-empty id per generation run")
        if not 1 <= max_concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"max_concurrency must be 1..{MAX_CONCURRENCY}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_concurrency = max_concurrency
        self._key = api_key if api_key is not None else api_key_from_env()
        self._transport = transport or _default_transport
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._requests = 0

    # -- introspection ---------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TeacherClient(model={self.model!r}, base_url={self.base_url!r}, "
            f"session_id={self.session_id!r}, key=<redacted>)"
        )

    @property
    def requests(self) -> int:
        """Every HTTP attempt made, retries included."""
        return self._requests

    def _count_request(self) -> None:
        with self._lock:
            self._requests += 1

    # -- the call --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "x-opencode-session": self.session_id,
            "Content-Type": "application/json",
            "User-Agent": "shopagent-teacher/0.1",
        }

    def chat(self, system: str, user: str) -> TeacherReply:
        """One completion. Raises TeacherUnavailable or TeacherParseError."""
        payload = {
            "model": self.model,
            "temperature": 0,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}{CHAT_PATH}"
        last = "no attempt was made"

        for attempt in range(1, self.max_attempts + 1):
            self._count_request()
            started = time.monotonic()
            try:
                status, raw = self._transport(url, self._headers(), body, self.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"network error ({type(exc).__name__})"
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise TeacherUnavailable(last) from exc

            if status == 200:
                latency = int((time.monotonic() - started) * 1000)
                return TeacherReply(
                    text=_extract_text(raw),
                    model=self.model,
                    attempts=attempt,
                    latency_ms=latency,
                    status=status,
                )

            last = f"HTTP {status}"
            if status in NO_RETRY_STATUSES:
                raise TeacherUnavailable(last)
            if status == 403 and attempt >= 2:
                raise TeacherUnavailable(last)
            if status == 429 or 500 <= status <= 599 or status == 403:
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise TeacherUnavailable(last)
            raise TeacherUnavailable(last)

        raise TeacherUnavailable(last)

    def _backoff(self, attempt: int) -> None:
        index = min(attempt - 1, len(BACKOFF_SECONDS) - 1)
        self._sleep(BACKOFF_SECONDS[index])

    # -- fan-out ---------------------------------------------------------

    def map(self, prompts: list[tuple[str, str]]) -> list[TeacherReply]:
        """Run prompts 8-at-a-time at most, returning replies in input order."""
        if not prompts:
            return []
        workers = min(self.max_concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda pair: self.chat(*pair), prompts))
