from __future__ import annotations

import unittest
from typing import Any

from airdeck.budget.counters import BudgetCounters
from airdeck.budget.limiter import BudgetLimiter, BudgetLimits
from airdeck.overshoot.client import OvershootAPIError
from airdeck.overshoot.inference import InferenceScheduler, build_latest_frame_payload


class FakeCompletionClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {"choices": [{"message": {"content": self.content}}]}


class FailingCompletionClient:
    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = payload
        raise OvershootAPIError(402, "credits exhausted")


class InferenceTests(unittest.TestCase):
    def test_latest_frame_payload_uses_ovs_reference(self) -> None:
        payload = build_latest_frame_payload(model_id="model", stream_id="stream-1")

        content = payload["messages"][0]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "ovs://streams/stream-1?frame_index=-1")
        self.assertEqual(payload["temperature"], 0)

    def test_scheduler_enforces_rate_and_parses_response(self) -> None:
        now = 0.0
        scheduler = scheduler_for(
            FakeCompletionClient(
                '{"gesture":"OPEN_PALM","confidence":0.91,"hand_visible":true,'
                '"description":"Open hand"}'
            ),
            clock=lambda: now,
        )

        first = scheduler.try_infer(stream_id="stream-1", model_id="model")
        second = scheduler.try_infer(stream_id="stream-1", model_id="model")

        self.assertEqual(first.status, "OK")
        self.assertEqual(first.gesture.gesture, "OPEN_PALM")
        self.assertEqual(second.status, "SKIPPED")
        self.assertEqual(second.skipped_reason, "rate limited")

    def test_no_overlapping_completion_requests(self) -> None:
        scheduler = scheduler_for(FakeCompletionClient("{}"), clock=lambda: 0.0)
        scheduler._in_flight = True

        result = scheduler.try_infer(stream_id="stream-1", model_id="model")

        self.assertEqual(result.status, "SKIPPED")
        self.assertEqual(result.skipped_reason, "request already in flight")

    def test_http_402_disables_further_requests(self) -> None:
        scheduler = scheduler_for(FailingCompletionClient(), clock=lambda: 0.0)

        result = scheduler.try_infer(stream_id="stream-1", model_id="model")

        self.assertEqual(result.status, "CREDITS_EXHAUSTED")
        self.assertTrue(scheduler._limiter.hard_stopped)


def scheduler_for(client: Any, *, clock: Any) -> InferenceScheduler:
    limits = BudgetLimits(
        max_completion_requests=80,
        max_session_minutes=5,
        max_requests_per_minute=60,
        max_inference_hz=2.0,
    )
    counters = BudgetCounters(started_at_monotonic=0.0)
    return InferenceScheduler(
        client,
        counters,
        BudgetLimiter(limits),
        max_completion_requests=80,
        max_inference_hz=2.0,
        default_inference_hz=0.75,
        clock=clock,
    )


if __name__ == "__main__":
    unittest.main()
