from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any


GESTURE_LABELS: tuple[str, ...] = (
    "NO_GESTURE",
    "OPEN_PALM",
    "CLOSED_FIST",
    "POINT_UP",
    "POINT_DOWN",
    "THUMB_LEFT",
    "THUMB_RIGHT",
    "TWO_FINGERS",
    "HAND_TRACKED",
)


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class GestureFeedback:
    gesture: str
    confidence: float
    hand_visible: bool
    motion_score: float
    description: str
    timestamp_monotonic: float
    bounding_box: BoundingBox | None = None
    landmarks: tuple[LandmarkPoint, ...] = ()


class LocalGestureFeedbackAnalyzer:
    """MediaPipe hand-landmark detector for local V1 UI feedback.

    This local path exists only so the prototype can display honest hand/gesture feedback before
    Overshoot inference is connected. It does not authorize command execution.
    """

    def __init__(self, *, history_size: int = 4) -> None:
        self._history_size = history_size
        self._stable_gesture = "NO_GESTURE"
        self._candidate_gesture = "NO_GESTURE"
        self._candidate_count = 0
        self._hands: Any | None = None
        self._mp_hands: Any | None = None
        self._init_error: str | None = None

    def analyze(self, frame: Any) -> GestureFeedback:
        now = time.monotonic()
        hands = self._get_hands()
        if hands is None:
            return _empty_feedback(self._init_error or "MediaPipe hand detector unavailable", now)

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return _empty_feedback("OpenCV unavailable", now)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True

        if not result.multi_hand_landmarks:
            self._stable_gesture = "NO_GESTURE"
            self._candidate_gesture = "NO_GESTURE"
            self._candidate_count = 0
            return _empty_feedback("No hand detected", now)

        hand_landmarks = result.multi_hand_landmarks[0]
        points = tuple(
            LandmarkPoint(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks.landmark
        )
        box = bounding_box_from_landmarks(points, frame_width=frame.shape[1], frame_height=frame.shape[0])
        raw_gesture, raw_confidence, description = classify_landmarks(points)
        score = _hand_score(result)
        confidence = min(max((raw_confidence * 0.72) + (score * 0.28), 0.0), 1.0)

        gesture = self._stabilize(raw_gesture)
        if gesture != raw_gesture:
            confidence = max(confidence - 0.12, 0.2)
            description = f"Stabilizing {raw_gesture.lower().replace('_', ' ')}"

        return GestureFeedback(
            gesture=gesture,
            confidence=confidence,
            hand_visible=True,
            motion_score=box_area_score(box, frame_width=frame.shape[1], frame_height=frame.shape[0]),
            description=description,
            timestamp_monotonic=now,
            bounding_box=box,
            landmarks=points,
        )

    def _stabilize(self, raw_gesture: str) -> str:
        if raw_gesture == self._stable_gesture:
            self._candidate_gesture = raw_gesture
            self._candidate_count = 0
            return self._stable_gesture

        if raw_gesture != self._candidate_gesture:
            self._candidate_gesture = raw_gesture
            self._candidate_count = 1
        else:
            self._candidate_count += 1

        required = min(max(self._history_size // 2, 2), 3)
        if raw_gesture == "HAND_TRACKED":
            required = max(required, 8 if self._stable_gesture != "NO_GESTURE" else 4)
        if self._candidate_count >= required:
            self._stable_gesture = raw_gesture
            self._candidate_count = 0
        return self._stable_gesture

    def _get_hands(self) -> Any | None:
        if self._hands is not None:
            return self._hands
        if self._init_error is not None:
            return None
        configure_mediapipe_cache()
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError:
            self._init_error = "Install MediaPipe for local gesture detection"
            return None

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        return self._hands


def configure_mediapipe_cache() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = Path(tempfile.gettempdir()) / "airdeck-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


def classify_landmarks(points: tuple[LandmarkPoint, ...]) -> tuple[str, float, str]:
    if len(points) < 21:
        return "HAND_TRACKED", 0.2, "Incomplete hand landmarks"

    index = finger_direction(points, 5, 6, 8)
    middle = finger_direction(points, 9, 10, 12)
    ring = finger_direction(points, 13, 14, 16)
    pinky = finger_direction(points, 17, 18, 20)
    directions = (index, middle, ring, pinky)
    up_count = directions.count("up")
    down_count = directions.count("down")
    folded_count = directions.count("folded")
    thumb = thumb_direction(points)

    if thumb in {"left", "right"} and up_count == 0 and folded_count >= 2:
        return f"THUMB_{thumb.upper()}", 0.88, f"Thumb extended {thumb}"
    if index == "up" and middle == "up" and non_signal_fingers_lowered(points):
        return "TWO_FINGERS", 0.88, "Index and middle fingers extended"
    if up_count >= 4 and fingertip_spread(points) >= 0.13:
        return "OPEN_PALM", 0.92, "Four fingers extended and spread"
    if index == "up" and middle != "up" and ring != "up" and pinky != "up":
        return "POINT_UP", 0.86, "Index finger extended upward"
    if index == "down" and middle != "down" and ring != "down" and pinky != "down":
        return "POINT_DOWN", 0.82, "Index finger extended downward"
    if folded_count >= 4 and thumb in {"folded", "unknown"}:
        return "CLOSED_FIST", 0.88, "Fingers folded into palm"
    if up_count >= 3:
        return "OPEN_PALM", 0.74, "Mostly open hand"
    if folded_count >= 3:
        return "CLOSED_FIST", 0.72, "Mostly folded hand"
    if down_count >= 3:
        return "POINT_DOWN", 0.58, "Downward-oriented hand"
    return "HAND_TRACKED", 0.46, "Hand present, pose not classified"


def finger_direction(
    points: tuple[LandmarkPoint, ...],
    mcp_index: int,
    pip_index: int,
    tip_index: int,
) -> str:
    mcp = points[mcp_index]
    pip = points[pip_index]
    tip = points[tip_index]
    finger_length = distance(mcp, tip)
    joint_length = max(distance(mcp, pip), 0.001)
    extension_ratio = finger_length / joint_length
    straightness = segment_alignment(mcp, pip, tip)

    if distance(tip, mcp) < joint_length * 1.25 or abs(tip.y - mcp.y) < 0.09:
        return "folded"
    if (
        tip.y < pip.y - 0.035
        and tip.y < mcp.y - 0.075
        and extension_ratio > 1.55
        and straightness > 0.35
    ):
        return "up"
    if (
        tip.y > pip.y + 0.035
        and tip.y > mcp.y + 0.075
        and extension_ratio > 1.55
        and straightness > 0.35
    ):
        return "down"
    return "neutral"


def thumb_direction(points: tuple[LandmarkPoint, ...]) -> str:
    wrist = points[0]
    mcp = points[2]
    ip = points[3]
    tip = points[4]
    horizontal = tip.x - mcp.x
    vertical = abs(tip.y - wrist.y)

    if abs(horizontal) > 0.11 and abs(tip.x - ip.x) > 0.04 and vertical < 0.24:
        return "right" if horizontal > 0 else "left"
    if distance(tip, mcp) < distance(points[1], mcp) * 1.4:
        return "folded"
    return "unknown"


def fingertip_spread(points: tuple[LandmarkPoint, ...]) -> float:
    tips = (points[8], points[12], points[16], points[20])
    xs = [point.x for point in tips]
    return max(xs) - min(xs)


def non_signal_fingers_lowered(points: tuple[LandmarkPoint, ...]) -> bool:
    raised_pair_tip_y = max(points[8].y, points[12].y)
    return points[16].y > raised_pair_tip_y + 0.12 and points[20].y > raised_pair_tip_y + 0.12


def bounding_box_from_landmarks(
    points: tuple[LandmarkPoint, ...],
    *,
    frame_width: int,
    frame_height: int,
    padding: int = 20,
) -> BoundingBox:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    min_x = max(int(min(xs) * frame_width) - padding, 0)
    min_y = max(int(min(ys) * frame_height) - padding, 0)
    max_x = min(int(max(xs) * frame_width) + padding, frame_width - 1)
    max_y = min(int(max(ys) * frame_height) + padding, frame_height - 1)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max(max_x - min_x, 1),
        height=max(max_y - min_y, 1),
    )


def box_area_score(box: BoundingBox, *, frame_width: int, frame_height: int) -> float:
    frame_area = max(frame_width * frame_height, 1)
    return min(max((box.width * box.height) / frame_area * 3.0, 0.0), 1.0)


def distance(a: LandmarkPoint, b: LandmarkPoint) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def segment_alignment(a: LandmarkPoint, b: LandmarkPoint, c: LandmarkPoint) -> float:
    first = (b.x - a.x, b.y - a.y, b.z - a.z)
    second = (c.x - b.x, c.y - b.y, c.z - b.z)
    first_length = max((first[0] ** 2 + first[1] ** 2 + first[2] ** 2) ** 0.5, 0.001)
    second_length = max((second[0] ** 2 + second[1] ** 2 + second[2] ** 2) ** 0.5, 0.001)
    dot = (first[0] * second[0]) + (first[1] * second[1]) + (first[2] * second[2])
    return dot / (first_length * second_length)


def _hand_score(result: Any) -> float:
    try:
        return float(result.multi_handedness[0].classification[0].score)
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0.75


def _empty_feedback(description: str, timestamp: float) -> GestureFeedback:
    return GestureFeedback(
        gesture="NO_GESTURE",
        confidence=0.0,
        hand_visible=False,
        motion_score=0.0,
        description=description,
        timestamp_monotonic=timestamp,
        bounding_box=None,
        landmarks=(),
    )
