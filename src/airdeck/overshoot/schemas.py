from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


ALLOWED_GESTURES = frozenset(
    {
        "OPEN_PALM",
        "THUMB_RIGHT",
        "THUMB_LEFT",
        "POINT_UP",
        "POINT_DOWN",
        "CLOSED_FIST",
        "TWO_FINGERS",
        "NO_GESTURE",
        "UNCERTAIN",
    }
)
NEUTRAL_GESTURES = frozenset({"NO_GESTURE", "UNCERTAIN"})


class GestureParseError(ValueError):
    """Raised when model output fails the strict gesture response contract."""


@dataclass(frozen=True)
class GestureInference:
    gesture: str
    confidence: float
    hand_visible: bool
    description: str
    epoch: int
    latency_ms: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return self.gesture in NEUTRAL_GESTURES


def parse_chat_completion_content(payload: dict[str, Any]) -> str:
    try:
        choices = payload["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GestureParseError("chat completion response is missing message content") from exc
    if not isinstance(content, str):
        raise GestureParseError("chat completion content must be a string")
    return content


def parse_gesture_content(
    content: str,
    *,
    response_epoch: int,
    current_epoch: int,
    latency_ms: float = 0.0,
) -> GestureInference:
    if response_epoch != current_epoch:
        raise GestureParseError("stale response epoch")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GestureParseError("model response must be valid JSON only") from exc
    return parse_gesture_payload(payload, response_epoch=response_epoch, latency_ms=latency_ms)


def parse_gesture_payload(
    payload: dict[str, Any],
    *,
    response_epoch: int,
    latency_ms: float = 0.0,
) -> GestureInference:
    if not isinstance(payload, dict):
        raise GestureParseError("gesture response must be a JSON object")
    if set(payload) != {"gesture", "confidence", "hand_visible", "description"}:
        raise GestureParseError("gesture response has missing or extra fields")

    gesture = payload["gesture"]
    confidence = payload["confidence"]
    hand_visible = payload["hand_visible"]
    description = payload["description"]

    if not isinstance(gesture, str) or gesture not in ALLOWED_GESTURES:
        raise GestureParseError("gesture is not in the allowed enum")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise GestureParseError("confidence must be a number")
    confidence_float = float(confidence)
    if confidence_float < 0.0 or confidence_float > 1.0:
        raise GestureParseError("confidence must be between 0 and 1")
    if not isinstance(hand_visible, bool):
        raise GestureParseError("hand_visible must be boolean")
    if not isinstance(description, str) or not description.strip():
        raise GestureParseError("description must be a non-empty string")

    if gesture in NEUTRAL_GESTURES and hand_visible and confidence_float >= 0.8:
        raise GestureParseError("neutral gestures cannot be high-confidence visible hands")

    return GestureInference(
        gesture=gesture,
        confidence=confidence_float,
        hand_visible=hand_visible,
        description=description.strip(),
        epoch=response_epoch,
        latency_ms=latency_ms,
    )
