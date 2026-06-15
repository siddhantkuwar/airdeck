from __future__ import annotations

from dataclasses import dataclass

from airdeck.budget.counters import BudgetCounters


@dataclass(frozen=True)
class BudgetLimits:
    max_completion_requests: int
    max_session_minutes: int
    max_requests_per_minute: int
    max_inference_hz: float
    warning_fraction: float = 0.70
    disable_fraction: float = 0.90


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    status: str
    reason: str
    fraction_used: float


class BudgetLimiter:
    def __init__(self, limits: BudgetLimits) -> None:
        self._limits = limits
        self._inference_disabled = False
        self._hard_stopped = False

    @property
    def inference_disabled(self) -> bool:
        return self._inference_disabled

    @property
    def hard_stopped(self) -> bool:
        return self._hard_stopped

    def disable_inference(self, reason: str = "disabled") -> BudgetDecision:
        self._inference_disabled = True
        return BudgetDecision(False, "INFERENCE_DISABLED", reason, 1.0)

    def resume_inference(self) -> None:
        if not self._hard_stopped:
            self._inference_disabled = False

    def stop_for_http_status(self, status: int) -> BudgetDecision | None:
        if status in {401, 402, 403}:
            self._hard_stopped = True
            self._inference_disabled = True
            label = "CREDITS_EXHAUSTED" if status == 402 else "AUTH_STOPPED"
            return BudgetDecision(False, label, f"fatal HTTP {status}", 1.0)
        return None

    def can_request(self, counters: BudgetCounters, *, now: float) -> BudgetDecision:
        if self._hard_stopped:
            return BudgetDecision(False, "HARD_STOP", "budget hard stop active", 1.0)
        if self._inference_disabled:
            return BudgetDecision(False, "INFERENCE_DISABLED", "inference disabled", 1.0)

        request_fraction = _fraction(
            counters.total_completion_requests,
            self._limits.max_completion_requests,
        )
        session_fraction = _fraction(
            counters.session_seconds(now),
            self._limits.max_session_minutes * 60,
        )
        rpm_fraction = _fraction(
            counters.requests_per_minute(now),
            self._limits.max_requests_per_minute,
        )
        fraction_used = max(request_fraction, session_fraction, rpm_fraction)

        if fraction_used >= 1.0:
            self._hard_stopped = True
            self._inference_disabled = True
            return BudgetDecision(False, "HARD_STOP", "budget limit reached", fraction_used)
        if fraction_used >= self._limits.disable_fraction:
            self._inference_disabled = True
            return BudgetDecision(
                False,
                "INFERENCE_DISABLED",
                "budget reached automatic disable threshold",
                fraction_used,
            )
        if fraction_used >= self._limits.warning_fraction:
            return BudgetDecision(True, "BUDGET_WARNING", "budget warning threshold reached", fraction_used)
        return BudgetDecision(True, "OK", "within budget", fraction_used)


def _fraction(value: float, limit: float) -> float:
    if limit <= 0:
        return 1.0
    return min(max(value / limit, 0.0), 1.0)
