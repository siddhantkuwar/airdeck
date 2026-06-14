from __future__ import annotations

from typing import Any

from airdeck.camera.gesture_feedback import GestureFeedback


def mirror_frame(frame: Any) -> Any:
    try:
        return frame[:, ::-1].copy()
    except (AttributeError, TypeError, IndexError):
        return frame


def frame_to_ppm(frame: Any, *, mirror: bool = True) -> bytes:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2:
        raise ValueError("Preview unavailable for this frame type")

    preview_frame = mirror_frame(frame) if mirror else frame
    height, width = int(shape[0]), int(shape[1])
    if len(shape) == 2:
        rgb_frame = preview_frame
    elif len(shape) == 3 and shape[2] >= 3:
        rgb_frame = preview_frame[:, :, 2::-1]
    else:
        raise ValueError("Preview unavailable for this frame shape")

    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb_frame.tobytes()


def draw_feedback_overlay(frame: Any, feedback: GestureFeedback) -> Any:
    if not feedback.hand_visible or feedback.bounding_box is None:
        return frame
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return frame

    output = frame.copy()
    box = feedback.bounding_box
    color = (66, 245, 152)
    cv2.rectangle(output, (box.x, box.y), (box.x + box.width, box.y + box.height), color, 3)
    draw_landmarks(output, feedback, color=color)

    label = f"{feedback.gesture.replace('_', ' ')} {round(feedback.confidence * 100)}%"
    baseline_y = max(box.y - 14, 26)
    (text_width, text_height), _baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        2,
    )
    cv2.rectangle(
        output,
        (box.x, baseline_y - text_height - 12),
        (box.x + text_width + 14, baseline_y + 8),
        color,
        -1,
    )
    cv2.putText(
        output,
        label,
        (box.x + 7, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (5, 20, 12),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_landmarks(frame: Any, feedback: GestureFeedback, *, color: tuple[int, int, int]) -> None:
    if not feedback.landmarks:
        return
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in feedback.landmarks
    ]
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    )
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return
    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (23, 23, 23), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 2, color, -1, cv2.LINE_AA)


def resize_frame_to_fit(frame: Any, *, max_width: int, max_height: int) -> Any:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2 or max_width <= 0 or max_height <= 0:
        return frame
    height, width = int(shape[0]), int(shape[1])
    scale = min(max_width / width, max_height / height)
    if scale <= 0 or abs(scale - 1.0) < 0.04:
        return frame
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return frame
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
