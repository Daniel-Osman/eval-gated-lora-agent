"""The training app's composed-plan parser: one root object, nothing else.

`parse_plan` is what the load check calls on a composed generation, so it is also
the only place a free-text plan could sneak into composed mode. A list of tool
calls is a valid free-text plan and an invalid composed plan, and it is exactly
what the tuned model emitted on N-105 (see `out/modal-check.log`). These tests
pin the contract: the root value is one JSON object, its keys pass the project's
own validator, and no text sits before or after it.
"""

from __future__ import annotations

import unittest

from train.modal_app import parse_plan

PLAN = '{"action": "refund", "amount_cents": 2100, "order_id": "ORD-1059"}'
FREE_TEXT_LIST = (
    '[{"args": {"order_id": "ORD-1059"}, "name": "get_order"}, '
    '{"args": {"amount_cents": 2100, "order_id": "ORD-1059"}, "name": "refund_order"}]'
)


class RootObjectTests(unittest.TestCase):
    def test_one_root_object_passes(self):
        plan, verdict = parse_plan(PLAN)

        self.assertEqual(verdict, "ok")
        self.assertEqual(plan, {"action": "refund", "amount_cents": 2100, "order_id": "ORD-1059"})

    def test_object_followed_by_the_stop_token_passes(self):
        plan, verdict = parse_plan(f"{PLAN}<|im_end|>")

        self.assertEqual(verdict, "ok")
        self.assertEqual(plan["order_id"], "ORD-1059")


class FreeTextPlanTests(unittest.TestCase):
    def test_free_text_list_at_the_root_is_refused(self):
        plan, verdict = parse_plan(FREE_TEXT_LIST)

        self.assertIsNone(plan)
        self.assertIn("not an object at the root", verdict)

    def test_free_text_list_inside_a_fence_is_refused(self):
        plan, verdict = parse_plan(f"```json\n{FREE_TEXT_LIST}\n```")

        self.assertIsNone(plan)
        self.assertIn("not an object at the root", verdict)

    def test_prose_before_a_fence_is_refused(self):
        plan, verdict = parse_plan(f"Here is the plan: ```json\n{PLAN}\n```")

        self.assertIsNone(plan)
        self.assertIn("outside the fence", verdict)

    def test_text_after_a_fence_is_refused(self):
        plan, verdict = parse_plan(f"```json\n{PLAN}\n``` trailing")

        self.assertIsNone(plan)
        self.assertIn("outside the fence", verdict)


class ContractTests(unittest.TestCase):
    def test_an_extra_key_is_refused(self):
        plan, verdict = parse_plan(
            '{"action": "refund", "amount_cents": 2100, "order_id": "ORD-1059", "message": "done"}'
        )

        self.assertIsNone(plan)
        self.assertIn("composed-plan validator refused", verdict)

    def test_trailing_text_is_refused(self):
        plan, verdict = parse_plan(f"{PLAN} and that is the plan.")

        self.assertIsNone(plan)
        self.assertIn("trailing text after the plan", verdict)

    def test_leading_prose_is_refused(self):
        plan, verdict = parse_plan(f"Here is the plan: {PLAN}")

        self.assertIsNone(plan)
        self.assertIn("not one JSON value at the root", verdict)


if __name__ == "__main__":
    unittest.main()
