"""The deployed endpoint's contract, and the two bugs that would be unforgivable.

A fake tokenizer and a fake model are enough to run the whole loop: the prompt
shaping, the plan parsers, the tool calls, the gate and the response body are all
deterministic once generation is replaced. What is pinned here:

* exactly one of `scenario` / `ticket`, and a 400 that names the cap;
* presets come from the tracked `bench/cases/normal-ext.jsonl`, never `data/sft`;
* **one** gate attempt: a composed refund appears once in the call log, and the
  module never reaches for `run_gated`, whose retry loop would refund twice;
* free text is two generations and is labelled unproven;
* an over-long prompt is a 400, an over-long plan is refused before it runs;
* `Agent.serve` crosses the RPC wire as one JSON string, never a pickled dict: the
  caller's image has no torch, and a non-JSON value raises instead of serialising.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from serve import app as app_mod

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "serve" / "app.py"


def code_only() -> str:
    """The source with its module docstring removed.

    The docstring explains *why* the endpoint avoids `run_gated`, the subscription
    key and the git-ignored SFT file; the tests below are about the code, so they
    must not read those words as usage.
    """
    text = SOURCE.read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(text), clean=False) or ""
    return text.replace(docstring, "") if docstring else text


class FakeTokenizer:
    """Deterministic stand-in for the Qwen chat template.

    It renders turns in the shape the records use, counts a token per word, and
    keeps every prompt it was asked to tokenize so a test can look inside.
    """

    def __init__(self, tokens_per_word: int = 1):
        self.tokens_per_word = tokens_per_word
        self.prompts: list[str] = []
        self.messages: list[list[dict]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        self.messages.append(messages)
        parts = []
        for message in messages:
            body = message["content"]
            if message["role"] == "tool":
                body = f"<tool_response>\n{body}\n</tool_response>"
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>")
        # The real Qwen template ends the render with a newline after the last
        # turn; the fake must too, or a bug like the one this caught is invisible.
        return "\n".join(parts) + "\n"

    def __call__(self, text, add_special_tokens=False):
        self.prompts.append(text)
        return {"input_ids": list(range(len(text.split()) * self.tokens_per_word))}


class FakeModel:
    """Returns pre-written generations in order and records every call."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls: list[tuple[int, int]] = []

    def __call__(self, ids, max_new_tokens):
        self.calls.append((len(ids), max_new_tokens))
        if not self.replies:
            raise AssertionError("the endpoint asked for a generation it was not capped for")
        text = self.replies.pop(0)
        return {"text": text, "output_tokens": len(text.split())}

    @property
    def generations(self) -> int:
        return len(self.calls)


def presets() -> dict:
    return app_mod.load_presets(REPO / "bench")


def composed(selection: dict, model: FakeModel, tokenizer: FakeTokenizer | None = None) -> dict:
    return app_mod.run_request(selection, tokenizer=tokenizer or FakeTokenizer(), generate=model)


class RequestContractTest(unittest.TestCase):
    def setUp(self):
        self.presets = presets()

    def test_exactly_one_selection_field(self):
        both = {"scenario": "N-104", "ticket": "hello"}
        neither: dict = {}
        for payload in (both, neither):
            with self.subTest(payload=payload):
                with self.assertRaises(app_mod.RequestError) as caught:
                    app_mod.validate_request(payload, self.presets)
                self.assertIn("exactly one", str(caught.exception))

    def test_body_must_be_an_object(self):
        with self.assertRaises(app_mod.RequestError):
            app_mod.validate_request(["N-104"], self.presets)

    def test_unknown_scenario_lists_the_presets(self):
        with self.assertRaises(app_mod.RequestError) as caught:
            app_mod.validate_request({"scenario": "N-999"}, self.presets)
        text = str(caught.exception)
        self.assertIn("N-104", text)
        self.assertIn("N-81", text)

    def test_ticket_cap_names_the_number(self):
        long_ticket = "x" * (app_mod.MAX_TICKET_CHARS + 1)
        with self.assertRaises(app_mod.RequestError) as caught:
            app_mod.validate_request({"ticket": long_ticket}, self.presets)
        self.assertIn(str(app_mod.MAX_TICKET_CHARS), str(caught.exception))
        self.assertIn(str(len(long_ticket)), str(caught.exception))

    def test_ticket_at_the_cap_is_accepted(self):
        selection = app_mod.validate_request({"ticket": "x" * app_mod.MAX_TICKET_CHARS}, self.presets)
        self.assertEqual(selection["mode"], "free_text")

    def test_blank_ticket_is_refused(self):
        for ticket in ("", "   \n"):
            with self.subTest(ticket=ticket):
                with self.assertRaises(app_mod.RequestError):
                    app_mod.validate_request({"ticket": ticket}, self.presets)

    def test_preset_selection_is_composed(self):
        selection = app_mod.validate_request({"scenario": "N-104"}, self.presets)
        self.assertEqual(selection["mode"], "composed")
        self.assertEqual(selection["preset"]["id"], "N-104")


class PresetsTest(unittest.TestCase):
    def test_presets_are_the_frozen_composed_cases(self):
        loaded = presets()
        self.assertEqual(len(loaded), 20)
        for preset in loaded.values():
            self.assertTrue(preset["case"]["composed"], preset["id"])
            self.assertTrue(preset["ticket"].strip(), preset["id"])

    def test_training_seen_is_labelled_per_preset(self):
        loaded = presets()
        self.assertTrue(loaded["N-104"]["training_seen"])
        self.assertTrue(loaded["N-105"]["training_seen"])
        self.assertFalse(loaded["N-81"]["training_seen"])
        self.assertFalse(loaded["N-110"]["training_seen"])

    def test_preset_view_hides_the_expected_answer(self):
        view = app_mod.preset_view(presets()["N-104"])
        self.assertEqual(set(view), {"id", "title", "kind", "training_seen", "ticket"})
        self.assertNotIn("required_calls", view)

    def test_no_git_ignored_training_file_is_a_source(self):
        text = code_only()
        self.assertNotIn("data/sft", text)
        self.assertNotIn("sft.jsonl", text)
        self.assertIn("normal-ext.jsonl", repr(app_mod.PRESET_FILE))


class ComposedLoopTest(unittest.TestCase):
    def test_a_refund_runs_once_and_the_gate_approves(self):
        model = FakeModel('{"action": "refund", "order_id": "ORD-1023", "amount_cents": 2599}')
        result = composed({"mode": "composed", "preset": presets()["N-104"]}, model)

        self.assertEqual(model.generations, 1)
        self.assertEqual(result["timings"]["generations"], 1)
        self.assertEqual(result["verdict"]["verdict"], "APPROVED")
        self.assertEqual(result["reply"], "We refunded $25.99 for ORD-1023.")
        self.assertTrue(result["verdict"]["reply_committed"])
        self.assertEqual([call["tool"] for call in result["calls"]], ["refund_order"])
        self.assertTrue(result["scenario"]["matches_required_calls"])

    def test_one_attempt_only_no_second_refund(self):
        """The worst bug in the project: a retried composed plan refunds twice."""
        model = FakeModel('{"action": "refund", "order_id": "ORD-1023", "amount_cents": 2599}')
        result = composed({"mode": "composed", "preset": presets()["N-104"]}, model)
        refunds = [call for call in result["calls"] if call["tool"] == "refund_order"]
        self.assertEqual(len(refunds), 1)
        self.assertEqual(refunds[0]["result"]["data"]["refund"]["amount"], 2599)

    def test_a_refused_action_renders_the_refusal(self):
        model = FakeModel('{"action": "refund", "order_id": "ORD-1059", "amount_cents": 2100}')
        result = composed({"mode": "composed", "preset": presets()["N-105"]}, model)
        self.assertEqual(result["reply"], "We could not process a refund for ORD-1059.")
        self.assertFalse(result["calls"][0]["result"]["ok"])
        self.assertEqual(result["verdict"]["verdict"], "APPROVED")

    def test_unparseable_plan_is_rejected_without_running_anything(self):
        model = FakeModel("I am sorry, I cannot help with that.")
        result = composed({"mode": "composed", "preset": presets()["N-104"]}, model)
        self.assertEqual(result["verdict"]["verdict"], "REJECTED")
        self.assertEqual(result["verdict"]["codes"], ["SERVE_PLAN_UNPARSED"])
        self.assertEqual(result["calls"], [])
        self.assertEqual(result["reply"], "")

    def test_a_plan_naming_a_missing_order_is_refused_not_a_crash(self):
        model = FakeModel('{"action": "status", "order_id": "ORD-9999"}')
        result = composed({"mode": "composed", "preset": presets()["N-99"]}, model)
        self.assertEqual(result["verdict"]["codes"], ["SERVE_COMPOSE_FAILED"])
        self.assertIn("ORD-9999", result["verdict"]["reasons"][0])

    def test_every_response_says_phase_b_is_off(self):
        model = FakeModel('{"action": "status", "order_id": "ORD-1029"}')
        result = composed({"mode": "composed", "preset": presets()["N-99"]}, model)
        self.assertFalse(result["phase_b"]["enabled"])
        self.assertIn("key", result["phase_b"]["reason"])
        self.assertEqual(result["verdict"]["phase"], "A")

    def test_the_unproven_line_tells_seen_from_unseen(self):
        """N-104 trained on; N-110 never did. One line each, and neither borrows the other."""
        seen = composed(
            {"mode": "composed", "preset": presets()["N-104"]},
            FakeModel('{"action": "refund", "order_id": "ORD-1023", "amount_cents": 2599}'),
        )
        unseen = composed(
            {"mode": "composed", "preset": presets()["N-110"]},
            FakeModel(
                '{"action": "update_address", "address_id": "ADR-2018", '
                '"line1": "14 Cedar Lane", "city": "Byblos"}'
            ),
        )
        self.assertEqual(len(seen["unproven"]), 1)
        self.assertIn("training-seen", seen["unproven"][0])
        self.assertNotIn("never trained on", seen["unproven"][0])
        self.assertEqual(len(unseen["unproven"]), 1)
        self.assertIn("never trained on", unseen["unproven"][0])
        self.assertNotIn("training-seen", unseen["unproven"][0])


class FreeTextLoopTest(unittest.TestCase):
    def plan(self, calls: list[dict]) -> str:
        return json.dumps(calls)

    def test_two_generations_and_the_gate_approves(self):
        tokenizer = FakeTokenizer()
        model = FakeModel(
            self.plan([{"name": "refund_order", "args": {"order_id": "ORD-1023", "amount_cents": 2599}}]),
            "We refunded $25.99 for ORD-1023.",
        )
        result = app_mod.run_request(
            {"mode": "free_text", "ticket": "Refund $25.99 for ORD-1023, please."},
            tokenizer=tokenizer,
            generate=model,
        )
        self.assertEqual(model.generations, 2)
        self.assertEqual(result["timings"]["generations"], 2)
        self.assertEqual(result["verdict"]["verdict"], "APPROVED")
        self.assertEqual(result["reply"], "We refunded $25.99 for ORD-1023.")
        self.assertEqual([call["tool"] for call in result["calls"]], ["refund_order"])
        # The second prompt carries the tool turn the model was trained to read.
        self.assertIn("<tool_response>", tokenizer.prompts[-1])
        self.assertIn("refund_succeeded", tokenizer.prompts[-1])

    def test_a_lying_reply_is_rejected_by_the_gate(self):
        model = FakeModel(
            self.plan([{"name": "refund_order", "args": {"order_id": "ORD-1023", "amount_cents": 2599}}]),
            "We refunded $99.00 for ORD-1023.",
        )
        result = app_mod.run_request(
            {"mode": "free_text", "ticket": "Refund $25.99 for ORD-1023."},
            tokenizer=FakeTokenizer(),
            generate=model,
        )
        self.assertEqual(result["verdict"]["verdict"], "REJECTED")
        self.assertTrue(result["verdict"]["codes"])
        self.assertFalse(result["verdict"]["reply_committed"])

    def test_free_text_is_labelled_unproven(self):
        model = FakeModel(self.plan([{"name": "get_order", "args": {"order_id": "ORD-1001"}}]), "It is paid.")
        result = app_mod.run_request(
            {"mode": "free_text", "ticket": "What is ORD-1001's status?"},
            tokenizer=FakeTokenizer(),
            generate=model,
        )
        self.assertTrue(result["unproven"])
        self.assertIn("unmeasured", " ".join(result["unproven"]))

    def test_a_plan_object_instead_of_a_list_is_refused(self):
        model = FakeModel('{"action": "refund", "order_id": "ORD-1023", "amount_cents": 2599}')
        result = app_mod.run_request(
            {"mode": "free_text", "ticket": "Refund $25.99 for ORD-1023."},
            tokenizer=FakeTokenizer(),
            generate=model,
        )
        self.assertEqual(result["verdict"]["codes"], ["SERVE_PLAN_UNPARSED"])
        self.assertIn("not a list", result["verdict"]["reasons"][0])

    def test_send_reply_is_not_a_plan_tool(self):
        model = FakeModel(self.plan([{"name": "send_reply", "args": {"ticket_id": "x", "message": "done"}}]))
        result = app_mod.run_request(
            {"mode": "free_text", "ticket": "Refund it."},
            tokenizer=FakeTokenizer(),
            generate=model,
        )
        self.assertEqual(result["verdict"]["codes"], ["SERVE_PLAN_UNPARSED"])
        self.assertIn("not a plan tool", result["verdict"]["reasons"][0])

    def test_a_plan_over_the_call_cap_is_refused_before_any_call_runs(self):
        calls = [{"name": "get_order", "args": {"order_id": "ORD-1001"}}] * (app_mod.MAX_TOOL_CALLS + 1)
        model = FakeModel(self.plan(calls))
        result = app_mod.run_request(
            {"mode": "free_text", "ticket": "Look at everything."},
            tokenizer=FakeTokenizer(),
            generate=model,
        )
        self.assertEqual(result["verdict"]["codes"], ["SERVE_PLAN_TOO_LONG"])
        self.assertEqual(result["calls"], [])
        self.assertEqual(model.generations, 1)

    def test_an_over_long_prompt_is_a_400(self):
        model = FakeModel("never used")
        with self.assertRaises(app_mod.RequestError) as caught:
            app_mod.run_request(
                {"mode": "free_text", "ticket": "hello"},
                tokenizer=FakeTokenizer(tokens_per_word=50),
                generate=model,
            )
        self.assertIn("max_model_len", str(caught.exception))
        self.assertEqual(model.generations, 0)

    def test_generation_caps_are_the_declared_ones(self):
        model = FakeModel(self.plan([{"name": "get_order", "args": {"order_id": "ORD-1001"}}]), "It is paid.")
        app_mod.run_request(
            {"mode": "free_text", "ticket": "status?"}, tokenizer=FakeTokenizer(), generate=model
        )
        self.assertEqual([cap for _, cap in model.calls], [app_mod.MAX_NEW_TOKENS_PLAN, app_mod.MAX_NEW_TOKENS_REPLY])

    def test_caps_are_pinned_to_their_measured_values(self):
        # 64/128/256 new tokens and 6 tool calls are the operating point measured
        # against max_model_len 4096: a wider cap eats the KV headroom, and a plan
        # longer than six calls is not one support action.
        self.assertEqual(app_mod.MAX_NEW_TOKENS_COMPOSED, 64)
        self.assertEqual(app_mod.MAX_NEW_TOKENS_PLAN, 128)
        self.assertEqual(app_mod.MAX_NEW_TOKENS_REPLY, 256)
        self.assertEqual(app_mod.MAX_TOOL_CALLS, 6)

    def test_tool_turn_shape_matches_training(self):
        turns = app_mod.tool_messages([{"tool": "get_order", "args": {"order_id": "ORD-1001"}, "result": {"ok": True}}])
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["role"], "tool")
        payload = json.loads(turns[0]["content"])
        self.assertEqual(sorted(payload), ["args", "result", "tool"])
        self.assertEqual(turns[0]["content"], json.dumps(payload, sort_keys=True))


class ServeDoorTest(unittest.TestCase):
    """`Agent.serve` is the RPC door: JSON text out, never a pickled object.

    The demo app's image has fastapi and httpx and no torch, so the agent's own
    return value failed to unpickle on the caller side (`out/rpc_probe.py`). The
    door returns one JSON string; `json.dumps` refuses anything that is not plain
    JSON, so the wire format cannot drift from the contract.
    """

    @staticmethod
    def door():
        """The undecorated `serve`: Modal 1.6 keeps it behind `_get_raw_f()`."""
        return app_mod.Agent._get_user_cls().__dict__["serve"]._get_raw_f()

    def call(self, answer):
        class Fake:
            """Only `_answer`, which is all `serve` reaches for."""

            def _answer(self, item):
                if isinstance(answer, Exception):
                    raise answer
                return answer

        return self.door()(Fake(), {"scenario": "N-104"})

    def test_a_success_is_one_json_string(self):
        wire = self.call({"mode": "composed", "verdict": {"verdict": "APPROVED"}})
        self.assertIsInstance(wire, str)
        self.assertEqual(
            json.loads(wire),
            {"status_code": 200, "body": {"mode": "composed", "verdict": {"verdict": "APPROVED"}}},
        )

    def test_a_refusal_is_a_value_not_a_raise(self):
        wire = self.call(app_mod.RequestError("unknown scenario"))
        self.assertEqual(
            json.loads(wire),
            {"status_code": 400, "body": {"detail": "unknown scenario"}},
        )

    def test_a_non_json_value_raises_instead_of_serialising(self):
        with self.assertRaises(TypeError):
            self.call({"x": object()})


class ModuleHygieneTest(unittest.TestCase):
    def test_no_model_or_web_import_at_module_level(self):
        """`modal deploy` and the test run import this file on Windows."""
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        heads = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                heads.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                heads.add(node.module.split(".")[0])
        for heavy in ("torch", "vllm", "fastapi", "transformers"):
            self.assertNotIn(heavy, heads, heavy)

    def test_no_retry_loop_around_the_gate(self):
        text = code_only()
        self.assertNotIn("run_gated", text)
        self.assertNotIn("for attempt", text)
        self.assertNotIn("MAX_ATTEMPTS", text)
        self.assertEqual(text.count("decision = evaluate("), 2)  # once per mode

    def test_volume_is_read_only_and_no_secret_is_mounted(self):
        text = code_only()
        self.assertIn("volume.read_only()", text)
        self.assertNotIn("modal.Secret", text)
        self.assertNotIn("API_KEY", text)

    def test_health_touches_no_model(self):
        payload = app_mod.health_payload(REPO / "bench")
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["engine"], "not loaded")
        self.assertEqual(len(payload["presets"]), 20)
        self.assertEqual(payload["phase_b"], "off")

    def test_composed_prompt_carries_the_composed_system_prompt(self):
        from shopagent.sft import COMPOSED_SYSTEM_PROMPT

        messages = app_mod.composed_messages("Refund $21.00 for ORD-1059.")
        self.assertEqual(messages[0]["content"], COMPOSED_SYSTEM_PROMPT)
        self.assertEqual(messages[-1], {"role": "user", "content": "Refund $21.00 for ORD-1059."})

    def test_the_assistant_prefix_is_the_training_shape(self):
        self.assertEqual(app_mod.ASSISTANT_PREFIX, "<|im_start|>assistant\n thinking\n\n</think>\n\n")
        prompt = app_mod.render_prompt(FakeTokenizer(), app_mod.free_text_messages("hi"))
        self.assertTrue(prompt.endswith(app_mod.ASSISTANT_PREFIX))

    def test_a_template_that_ends_with_a_newline_still_renders(self):
        """The deployed 500: the real template ends `<|im_end|>\\n`, not `<|im_end|>`."""
        prompt = app_mod.render_prompt(FakeTokenizer(), app_mod.composed_messages("Refund ORD-1023."))
        self.assertIn("<|im_end|>\n<|im_start|>assistant\n", prompt)
        self.assertTrue(prompt.endswith(app_mod.ASSISTANT_PREFIX))

    def test_a_template_that_closes_nothing_is_refused(self):
        class Unclosed(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
                return "no turn markers here"

        with self.assertRaises(RuntimeError):
            app_mod.render_prompt(Unclosed(), app_mod.composed_messages("x"))


if __name__ == "__main__":
    unittest.main()
