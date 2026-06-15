from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GestureCommand:
    gesture: str
    command: str
    label: str
    cooldown_seconds: float


GESTURE_COMMANDS: dict[str, GestureCommand] = {
    "OPEN_PALM": GestureCommand("OPEN_PALM", "TOGGLE_PLAYBACK", "Toggle playback", 1.5),
    "THUMB_RIGHT": GestureCommand("THUMB_RIGHT", "SEEK_FORWARD", "Seek forward", 1.0),
    "THUMB_LEFT": GestureCommand("THUMB_LEFT", "SEEK_BACKWARD", "Seek backward", 1.0),
    "POINT_DOWN": GestureCommand("POINT_DOWN", "SCROLL_DOWN", "Scroll down", 0.75),
    "POINT_UP": GestureCommand("POINT_UP", "SCROLL_UP", "Scroll up", 0.75),
    "CLOSED_FIST": GestureCommand("CLOSED_FIST", "STOP_SCROLL", "Stop scrolling", 0.75),
    "TWO_FINGERS": GestureCommand("TWO_FINGERS", "TOGGLE_LISTENING", "Toggle listening", 2.0),
}


def command_for_gesture(gesture: str) -> GestureCommand | None:
    return GESTURE_COMMANDS.get(gesture)
