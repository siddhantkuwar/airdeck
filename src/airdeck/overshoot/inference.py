from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from airdeck.budget.counters import BudgetCounters, BudgetSnapshot
from airdeck.budget.limiter import BudgetLimiter
from airdeck.overshoot.client import OvershootAPIError, OvershootClient
from airdeck.overshoot.schemas import (
    GestureInference,
    GestureParseError,
    parse_chat_completion_content,
    parse_gesture_content,
)


GESTURE_PROMPT = (
    "Return only compact JSON for the visible hand gesture. "
    "Allowed gesture values: OPEN_PALM, THUMB_RIGHT, THUMB_LEFT, POINT_UP, POINT_DOWN, "
    "CLOSED_FIST, TWO_FINGERS, NO_GESTURE, UNCERTAIN. "
    "Schema: {\"gesture\": string, \"confidence\": number 0..1, "
    "\"hand_visible\": boolean, \"description\": string}. No prose."
)


@dataclass(frozen=True)
class InferenceResult:
    status: str
    gesture: GestureInference | None
    latency_ms: float
    budget: BudgetSnapshot
    skipped_reason: str = ""
    http_status: int | None = None


class RateGate:
    def __init__(self, *, max_hz: float, default_hz: float) -> None:
        if max_hz <= 0 or default_hz <= 0:
            raise ValueError("inference rates must be positive")
        self._max_hz = max_hz
        self._default_hz = min(default_hz, max_hz)
        self._last_request_at: float | None = None

    @property
    def min_interval_seconds(self) -> float:
        return 1.0 / self._max_hz

    def can_request(self, now: float, *, target_hz: float | None = None) -> bool:
        if self._last_request_at is None:
            return True
        hz = min(target_hz or self._default_hz, self._max_hz)
        interval = max(1.0 / hz, self.min_interval_seconds)
        return now - self._last_request_at >= interval

    def mark_request(self, now: float) -> None:
        self._last_request_at = now


class InferenceScheduler:
    def __init__(
        self,
        client: OvershootClient,
        counters: BudgetCounters,
        limiter: BudgetLimiter,
        *,
        max_completion_requests: int,
        max_inference_hz: float,
        default_inference_hz: float,
        clock: callable = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client
        self._counters = counters
        self._limiter = limiter
        self._max_completion_requests = max_completion_requests
        self._rate_gate = RateGate(max_hz=max_inference_hz, default_hz=default_inference_hz)
        self._clock = clock
        self._logger = logger or logging.getLogger("airdeck.overshoot.inference")
        self._in_flight = False
        self._epoch = 0
        self._backoff_until = 0.0
        self._backoff_attempts = 0
        self._shutdown = False

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def shutdown(self) -> None:
        self._shutdown = True
        self._epoch += 1

    def next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    def try_infer(
        self,
        *,
        stream_id: str,
        model_id: str,
        target_hz: float | None = None,
    ) -> InferenceResult:
        now = self._clock()
        if self._shutdown:
            return self._skipped("shutdown")
        if self._in_flight:
            return self._skipped("request already in flight")
        if now < self._backoff_until:
            return self._skipped("API backoff active")
        budget_decision = self._limiter.can_request(self._counters, now=now)
        if not budget_decision.allowed:
            return self._skipped(budget_decision.status)
        if not self._rate_gate.can_request(now, target_hz=target_hz):
            return self._skipped("rate limited")

        request_epoch = self.next_epoch()
        payload = build_latest_frame_payload(model_id=model_id, stream_id=stream_id)
        self._in_flight = True
        self._rate_gate.mark_request(now)
        self._counters.record_request_started(now)
        started = self._clock()
        self._logger.info("overshoot_completion_started stream_id=%s model=%s", stream_id, model_id)
        try:
            response = self._client.chat_completion(payload)
            latency_ms = max((self._clock() - started) * 1000.0, 0.0)
            content = parse_chat_completion_content(response)
            gesture = parse_gesture_content(
                content,
                response_epoch=request_epoch,
                current_epoch=self._epoch,
                latency_ms=latency_ms,
            )
            self._counters.record_success(http_status=200, now=self._clock())
            self._backoff_attempts = 0
            self._logger.info(
                "overshoot_completion_parsed gesture=%s confidence=%.2f latency_ms=%.1f",
                gesture.gesture,
                gesture.confidence,
                latency_ms,
            )
            return InferenceResult(
                status="OK",
                gesture=gesture,
                latency_ms=latency_ms,
                budget=self._budget_snapshot(),
                http_status=200,
            )
        except GestureParseError as exc:
            self._counters.record_failure(http_status=200, now=self._clock())
            self._logger.warning("overshoot_completion_rejected reason=%s", exc)
            return InferenceResult(
                status="REJECTED",
                gesture=None,
                latency_ms=max((self._clock() - started) * 1000.0, 0.0),
                budget=self._budget_snapshot(),
                skipped_reason=str(exc),
                http_status=200,
            )
        except OvershootAPIError as exc:
            now = self._clock()
            self._counters.record_failure(http_status=exc.status, now=now)
            stop = self._limiter.stop_for_http_status(exc.status) if exc.status is not None else None
            if stop is not None:
                status = stop.status
                self.shutdown()
            elif exc.retryable:
                self._schedule_backoff(now)
                status = "API_BACKOFF"
            else:
                status = "ERROR"
            self._logger.warning("overshoot_completion_failed status=%s reason=%s", exc.status, exc)
            return InferenceResult(
                status=status,
                gesture=None,
                latency_ms=max((self._clock() - started) * 1000.0, 0.0),
                budget=self._budget_snapshot(),
                skipped_reason=str(exc),
                http_status=exc.status,
            )
        finally:
            self._in_flight = False

    def _schedule_backoff(self, now: float) -> None:
        self._backoff_attempts = min(self._backoff_attempts + 1, 3)
        base = 0.5 * (2 ** (self._backoff_attempts - 1))
        jitter = random.uniform(0.0, 0.15)
        self._backoff_until = now + base + jitter

    def _skipped(self, reason: str) -> InferenceResult:
        return InferenceResult(
            status="SKIPPED",
            gesture=None,
            latency_ms=0.0,
            budget=self._budget_snapshot(),
            skipped_reason=reason,
            http_status=self._counters.last_http_status,
        )

    def _budget_snapshot(self) -> BudgetSnapshot:
        return self._counters.snapshot(
            now=self._clock(),
            max_completion_requests=self._max_completion_requests,
        )


def build_latest_frame_payload(*, model_id: str, stream_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GESTURE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"ovs://streams/{stream_id}?frame_index=-1"},
                    },
                ],
            }
        ],
        "max_tokens": 80,
        "temperature": 0,
    }
