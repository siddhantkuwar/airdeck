from __future__ import annotations

import unittest

from airdeck.budget.counters import BudgetCounters
from airdeck.budget.limiter import BudgetLimiter, BudgetLimits


class BudgetTests(unittest.TestCase):
    def test_budget_warns_disables_and_hard_stops(self) -> None:
        counters = BudgetCounters(started_at_monotonic=0.0)
        limiter = BudgetLimiter(
            BudgetLimits(
                max_completion_requests=10,
                max_session_minutes=10,
                max_requests_per_minute=60,
                max_inference_hz=2.0,
            )
        )

        for index in range(7):
            counters.record_request_started(float(index))
        self.assertEqual(limiter.can_request(counters, now=7.0).status, "BUDGET_WARNING")

        for index in range(7, 9):
            counters.record_request_started(float(index))
        decision = limiter.can_request(counters, now=9.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.status, "INFERENCE_DISABLED")

    def test_http_402_hard_stops_inference(self) -> None:
        limiter = BudgetLimiter(
            BudgetLimits(
                max_completion_requests=80,
                max_session_minutes=5,
                max_requests_per_minute=60,
                max_inference_hz=2.0,
            )
        )

        decision = limiter.stop_for_http_status(402)

        self.assertIsNotNone(decision)
        self.assertTrue(limiter.hard_stopped)
        self.assertEqual(decision.status, "CREDITS_EXHAUSTED")

    def test_requests_per_minute_uses_last_sixty_seconds(self) -> None:
        counters = BudgetCounters(started_at_monotonic=0.0)
        counters.record_request_started(0.0)
        counters.record_request_started(10.0)
        counters.record_request_started(61.0)

        self.assertEqual(counters.requests_per_minute(61.0), 2.0)


if __name__ == "__main__":
    unittest.main()
