"""The serving helpers are copies, and a copy that drifts is a lie.

`serve/suite.py` repeats five functions from `train/modal_app.py` because a Modal
app cannot import another Modal app's module without dragging its image and its
Volume lookup along. Byte-identical behaviour is what matters, so this compares
syntax trees: same names, same defaults, same body. Line numbers are ignored, so
reformatting is free and a changed branch is not.

The functional table pins the two behaviours a duplicated helper could quietly
lose: the fence rules and the strict id-prefix guard.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from serve import suite

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "train" / "modal_app.py"
COPIED = ("_strip_stop_token", "parse_plan", "check_prompt", "training_target", "composed_verdict")


def function_dump(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node)
    raise AssertionError(f"{name} is not defined at module level in {path}")


class CopiedHelpersTest(unittest.TestCase):
    def test_copies_match_train_module(self):
        for name in COPIED:
            with self.subTest(name=name):
                self.assertEqual(function_dump(SOURCE, name), function_dump(REPO / "serve" / "suite.py", name))

    def test_constants_match_train_module(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in {"THINK_OPEN", "THINK_CLOSE", "ASSISTANT_TURN", "IM_END"}:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(
            values,
            {
                "THINK_OPEN": suite.THINK_OPEN,
                "THINK_CLOSE": suite.THINK_CLOSE,
                "ASSISTANT_TURN": suite.ASSISTANT_TURN,
                "IM_END": suite.IM_END,
            },
        )

    def test_parse_plan_refuses_what_it_refuses_today(self):
        plan = {"action": "refund", "order_id": "ORD-1059", "amount_cents": 2100}
        good = [
            json.dumps(plan),
            '```json\n{"action": "refund", "order_id": "ORD-1059", "amount_cents": 2100}\n```',
            json.dumps(plan) + "<|im_end|>",
        ]
        bad = [
            "",
            "prose before ```\n" + json.dumps(plan),
            "[{'action': 'refund'}]",
            json.dumps(plan) + " and then more",
            json.dumps({"action": "cancel", "order_id": "ORD-1059"}),
            json.dumps({**plan, "note": "sorry about that"}),
        ]
        for text in good:
            with self.subTest(good=text):
                parsed, why = suite.parse_plan(text)
                self.assertEqual(parsed, plan, why)
                self.assertEqual(why, "ok")
        for text in bad:
            with self.subTest(bad=text):
                parsed, why = suite.parse_plan(text)
                self.assertIsNone(parsed, why)

    def test_check_prompt_guards_the_id_prefix(self):
        text = (
            "<|im_start|>system\nbe brief<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n thinking\n\n</think>\n\n"
        )

        class Tokenizer:
            """The prompt really tokenizes to four ids; what the record claims may differ."""

            def __call__(self, prompt, add_special_tokens=False):
                return {"input_ids": [1, 2, 3, 4]}

        prompt = suite.check_prompt(text, "T-1", Tokenizer(), [1, 2, 3, 4])
        self.assertTrue(prompt.endswith("\n\n"))
        self.assertNotIn("```", prompt)
        self.assertFalse(prompt.rstrip().endswith("assistant"))

        with self.assertRaises(suite.RunError):
            suite.check_prompt(text, "T-1", Tokenizer(), [9, 9, 9, 9])


if __name__ == "__main__":
    unittest.main()
