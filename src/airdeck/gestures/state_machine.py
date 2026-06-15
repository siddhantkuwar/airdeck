from __future__ import annotations

from dataclasses import dataclass

from airdeck.gestures.mappings import GestureCommand, command_for_gesture
from airdeck.overshoot.schemas import GestureInference, NEUTRAL_GESTURES


@dataclass(frozen=True)
class GestureDecision:
    state: str
    accepted: bool
    command: GestureCommand | None
    candidate_gesture: str | None
    confirmation_count: int
    listening_enabled: bool
    reason: str


class GestureStateMachine:
    def __init__(
        self,
        *,
        confidence_threshold: float = 0.80,
        required_confirmations: int = 2,
        confirmation_window_seconds: float = 2.5,
        global_cooldown_seconds: float = 0.75,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._required_confirmations = required_confirmations
        self._confirmation_window_seconds = confirmation_window_seconds
        self._global_cooldown_seconds = global_cooldown_seconds
        self._listening_enabled = True
        self._state = "IDLE"
        self._candidate_gesture: str | None = None
        self._candidate_count = 0
        self._candidate_started_at = 0.0
        self._cooldown_until = 0.0
        self._held_gesture: str | None = None
        self._last_command_by_gesture: dict[str, float] = {}

    @property
    def listening_enabled(self) -> bool:
        return self._listening_enabled

    @property
    def confirmation_count(self) -> int:
        return self._candidate_count

    @property
    def state(self) -> str:
        return self._state

    def set_listening(self, enabled: bool) -> None:
        self._listening_enabled = enabled
        self._state = "IDLE" if enabled else "DISABLED"
        self._reset_candidate()

    def observe(self, inference: GestureInference, *, now: float) -> GestureDecision:
        if inference.gesture in NEUTRAL_GESTURES or not inference.hand_visible:
            self._held_gesture = None
            self._reset_candidate()
            self._state = "IDLE" if self._listening_enabled else "DISABLED"
            return self._decision(False, None, "neutral gesture")

        command = command_for_gesture(inference.gesture)
        if command is None:
            self._reset_candidate()
            self._state = "IDLE"
            return self._decision(False, None, "gesture has no command mapping")

        if not self._listening_enabled and inference.gesture != "TWO_FINGERS":
            self._state = "DISABLED"
            self._reset_candidate()
            return self._decision(False, None, "listening disabled")

        if inference.confidence < self._confidence_threshold:
            self._state = "IDLE"
            self._reset_candidate()
            return self._decision(False, None, "confidence below threshold")

        if now < self._cooldown_until:
            self._state = "COOLDOWN"
            return self._decision(False, command, "global cooldown active")

        if self._held_gesture == inference.gesture:
            self._state = "COOLDOWN"
            return self._decision(False, command, "held gesture suppressed")

        if self._is_command_in_cooldown(command, now):
            self._state = "COOLDOWN"
            return self._decision(False, command, "command cooldown active")

        self._track_candidate(inference.gesture, now)
        if self._candidate_count < self._required_confirmations:
            self._state = "CANDIDATE"
            return self._decision(False, command, "awaiting confirmation")

        self._state = "CONFIRMED"
        self._held_gesture = inference.gesture
        self._cooldown_until = now + self._global_cooldown_seconds
        self._last_command_by_gesture[inference.gesture] = now
        self._reset_candidate()
        if inference.gesture == "TWO_FINGERS":
            self._listening_enabled = not self._listening_enabled
            self._state = "DISABLED" if not self._listening_enabled else "IDLE"
        return self._decision(True, command, "gesture confirmed")

    def _track_candidate(self, gesture: str, now: float) -> None:
        expired = now - self._candidate_started_at > self._confirmation_window_seconds
        if self._candidate_gesture != gesture or expired:
            self._candidate_gesture = gesture
            self._candidate_count = 1
            self._candidate_started_at = now
            return
        self._candidate_count += 1

    def _is_command_in_cooldown(self, command: GestureCommand, now: float) -> bool:
        previous = self._last_command_by_gesture.get(command.gesture)
        if previous is None:
            return False
        return now - previous < command.cooldown_seconds

    def _reset_candidate(self) -> None:
        self._candidate_gesture = None
        self._candidate_count = 0
        self._candidate_started_at = 0.0

    def _decision(
        self,
        accepted: bool,
        command: GestureCommand | None,
        reason: str,
    ) -> GestureDecision:
        return GestureDecision(
            state=self._state,
            accepted=accepted,
            command=command if accepted else None,
            candidate_gesture=self._candidate_gesture,
            confirmation_count=self._candidate_count,
            listening_enabled=self._listening_enabled,
            reason=reason,
        )
