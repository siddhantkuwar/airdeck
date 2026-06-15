from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSnapshot:
    session_seconds: float
    stream_active_seconds: float
    total_completion_requests: int
    successful_completion_requests: int
    failed_completion_requests: int
    requests_per_minute: float
    current_inference_hz: float
    last_http_status: int | None
    remaining_completion_requests: int


class BudgetCounters:
    def __init__(self, *, started_at_monotonic: float = 0.0) -> None:
        self._started_at = started_at_monotonic
        self._stream_started_at: float | None = None
        self._stream_active_seconds = 0.0
        self._request_times: deque[float] = deque()
        self._total_completion_requests = 0
        self._successful_completion_requests = 0
        self._failed_completion_requests = 0
        self._last_http_status: int | None = None
        self._current_inference_hz = 0.0

    @property
    def total_completion_requests(self) -> int:
        return self._total_completion_requests

    @property
    def last_http_status(self) -> int | None:
        return self._last_http_status

    def start_stream(self, now: float) -> None:
        if self._stream_started_at is None:
            self._stream_started_at = now

    def stop_stream(self, now: float) -> None:
        if self._stream_started_at is not None:
            self._stream_active_seconds += max(now - self._stream_started_at, 0.0)
            self._stream_started_at = None

    def record_request_started(self, now: float) -> None:
        self._total_completion_requests += 1
        self._request_times.append(now)
        self._trim_request_window(now)
        self._current_inference_hz = self._calculate_recent_hz(now)

    def record_success(self, *, http_status: int, now: float) -> None:
        self._successful_completion_requests += 1
        self._last_http_status = http_status
        self._trim_request_window(now)

    def record_failure(self, *, http_status: int | None, now: float) -> None:
        self._failed_completion_requests += 1
        self._last_http_status = http_status
        self._trim_request_window(now)

    def requests_per_minute(self, now: float) -> float:
        self._trim_request_window(now)
        return float(len(self._request_times))

    def session_seconds(self, now: float) -> float:
        return max(now - self._started_at, 0.0)

    def stream_active_seconds(self, now: float) -> float:
        if self._stream_started_at is None:
            return self._stream_active_seconds
        return self._stream_active_seconds + max(now - self._stream_started_at, 0.0)

    def snapshot(self, *, now: float, max_completion_requests: int) -> BudgetSnapshot:
        return BudgetSnapshot(
            session_seconds=self.session_seconds(now),
            stream_active_seconds=self.stream_active_seconds(now),
            total_completion_requests=self._total_completion_requests,
            successful_completion_requests=self._successful_completion_requests,
            failed_completion_requests=self._failed_completion_requests,
            requests_per_minute=self.requests_per_minute(now),
            current_inference_hz=self._current_inference_hz,
            last_http_status=self._last_http_status,
            remaining_completion_requests=max(max_completion_requests - self._total_completion_requests, 0),
        )

    def _trim_request_window(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

    def _calculate_recent_hz(self, now: float) -> float:
        cutoff = now - 10.0
        recent = [timestamp for timestamp in self._request_times if timestamp >= cutoff]
        if len(recent) < 2:
            return 0.0
        span = max(now - recent[0], 0.001)
        return len(recent) / span

