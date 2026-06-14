from __future__ import annotations

import unittest

from airdeck.camera.gesture_feedback import (
    LandmarkPoint,
    LocalGestureFeedbackAnalyzer,
    bounding_box_from_landmarks,
    classify_landmarks,
)


class GestureFeedbackTests(unittest.TestCase):
    def test_open_palm_requires_extended_spread_fingers(self) -> None:
        gesture, confidence, _description = classify_landmarks(open_palm_points())

        self.assertEqual(gesture, "OPEN_PALM")
        self.assertGreater(confidence, 0.8)

    def test_closed_fist_requires_folded_fingers(self) -> None:
        gesture, confidence, _description = classify_landmarks(fist_points())

        self.assertEqual(gesture, "CLOSED_FIST")
        self.assertGreater(confidence, 0.7)

    def test_point_up_and_down_use_index_direction(self) -> None:
        up, _confidence, _description = classify_landmarks(point_up_points())
        down, _confidence, _description = classify_landmarks(point_down_points())

        self.assertEqual(up, "POINT_UP")
        self.assertEqual(down, "POINT_DOWN")

    def test_curled_index_below_knuckle_is_not_point_down(self) -> None:
        mapping = dict(enumerate(fist_points()))
        mapping.update(
            {
                5: LandmarkPoint(0.42, 0.50),
                6: LandmarkPoint(0.46, 0.62),
                8: LandmarkPoint(0.39, 0.66),
            }
        )

        gesture, _confidence, _description = classify_landmarks(
            tuple(mapping[index] for index in range(21))
        )

        self.assertEqual(gesture, "CLOSED_FIST")

    def test_two_fingers_requires_index_and_middle_only(self) -> None:
        gesture, _confidence, _description = classify_landmarks(two_fingers_points())

        self.assertEqual(gesture, "TWO_FINGERS")

    def test_thumb_direction_uses_screen_space(self) -> None:
        right, _confidence, _description = classify_landmarks(thumb_points("right"))
        left, _confidence, _description = classify_landmarks(thumb_points("left"))

        self.assertEqual(right, "THUMB_RIGHT")
        self.assertEqual(left, "THUMB_LEFT")

    def test_thumb_direction_wins_over_folded_fingers(self) -> None:
        gesture, confidence, _description = classify_landmarks(thumb_points("right"))

        self.assertEqual(gesture, "THUMB_RIGHT")
        self.assertGreater(confidence, 0.8)

    def test_two_fingers_allows_neutral_ring_and_pinky(self) -> None:
        mapping = dict(enumerate(two_fingers_points()))
        mapping.update(
            {
                16: LandmarkPoint(0.57, 0.53),
                20: LandmarkPoint(0.63, 0.56),
            }
        )

        gesture, _confidence, _description = classify_landmarks(
            tuple(mapping[index] for index in range(21))
        )

        self.assertEqual(gesture, "TWO_FINGERS")

    def test_bounding_box_from_landmarks_adds_padding(self) -> None:
        box = bounding_box_from_landmarks(open_palm_points(), frame_width=640, frame_height=480)

        self.assertGreater(box.width, 1)
        self.assertGreater(box.height, 1)
        self.assertGreaterEqual(box.x, 0)
        self.assertGreaterEqual(box.y, 0)

    def test_stabilizer_requires_repeated_gesture_before_switching(self) -> None:
        analyzer = LocalGestureFeedbackAnalyzer()

        self.assertEqual(analyzer._stabilize("OPEN_PALM"), "NO_GESTURE")
        self.assertEqual(analyzer._stabilize("CLOSED_FIST"), "NO_GESTURE")
        self.assertEqual(analyzer._stabilize("OPEN_PALM"), "NO_GESTURE")
        self.assertEqual(analyzer._stabilize("OPEN_PALM"), "OPEN_PALM")

    def test_stabilizer_does_not_flicker_to_hand_tracked(self) -> None:
        analyzer = LocalGestureFeedbackAnalyzer()
        analyzer._stable_gesture = "CLOSED_FIST"

        for _ in range(7):
            self.assertEqual(analyzer._stabilize("HAND_TRACKED"), "CLOSED_FIST")

        self.assertEqual(analyzer._stabilize("HAND_TRACKED"), "HAND_TRACKED")


def open_palm_points() -> tuple[LandmarkPoint, ...]:
    return points_from_map(
        {
            0: (0.50, 0.84), 1: (0.43, 0.76), 2: (0.36, 0.67), 3: (0.30, 0.58), 4: (0.24, 0.50),
            5: (0.41, 0.62), 6: (0.40, 0.45), 7: (0.39, 0.30), 8: (0.38, 0.18),
            9: (0.50, 0.60), 10: (0.50, 0.42), 11: (0.50, 0.27), 12: (0.50, 0.13),
            13: (0.59, 0.62), 14: (0.60, 0.46), 15: (0.61, 0.32), 16: (0.62, 0.20),
            17: (0.67, 0.67), 18: (0.69, 0.52), 19: (0.71, 0.40), 20: (0.73, 0.29),
        }
    )


def fist_points() -> tuple[LandmarkPoint, ...]:
    return points_from_map(
        {
            0: (0.50, 0.84), 1: (0.45, 0.75), 2: (0.43, 0.70), 3: (0.43, 0.67), 4: (0.44, 0.68),
            5: (0.42, 0.62), 6: (0.43, 0.66), 7: (0.44, 0.66), 8: (0.43, 0.62),
            9: (0.50, 0.60), 10: (0.50, 0.65), 11: (0.51, 0.65), 12: (0.50, 0.60),
            13: (0.58, 0.62), 14: (0.57, 0.66), 15: (0.56, 0.66), 16: (0.57, 0.62),
            17: (0.65, 0.67), 18: (0.63, 0.70), 19: (0.62, 0.70), 20: (0.63, 0.67),
        }
    )


def point_up_points() -> tuple[LandmarkPoint, ...]:
    mapping = dict(enumerate(fist_points()))
    mapping.update({5: LandmarkPoint(0.42, 0.62), 6: LandmarkPoint(0.41, 0.45), 8: LandmarkPoint(0.40, 0.18)})
    return tuple(mapping[index] for index in range(21))


def point_down_points() -> tuple[LandmarkPoint, ...]:
    mapping = dict(enumerate(fist_points()))
    mapping.update({5: LandmarkPoint(0.42, 0.42), 6: LandmarkPoint(0.42, 0.57), 8: LandmarkPoint(0.42, 0.78)})
    return tuple(mapping[index] for index in range(21))


def two_fingers_points() -> tuple[LandmarkPoint, ...]:
    mapping = dict(enumerate(point_up_points()))
    mapping.update({9: LandmarkPoint(0.50, 0.60), 10: LandmarkPoint(0.50, 0.42), 12: LandmarkPoint(0.50, 0.13)})
    return tuple(mapping[index] for index in range(21))


def thumb_points(direction: str) -> tuple[LandmarkPoint, ...]:
    mapping = dict(enumerate(fist_points()))
    if direction == "right":
        mapping.update({1: LandmarkPoint(0.52, 0.73), 2: LandmarkPoint(0.55, 0.70), 3: LandmarkPoint(0.64, 0.69), 4: LandmarkPoint(0.76, 0.68)})
    else:
        mapping.update({1: LandmarkPoint(0.48, 0.73), 2: LandmarkPoint(0.45, 0.70), 3: LandmarkPoint(0.36, 0.69), 4: LandmarkPoint(0.24, 0.68)})
    return tuple(mapping[index] for index in range(21))


def points_from_map(mapping: dict[int, tuple[float, float]]) -> tuple[LandmarkPoint, ...]:
    return tuple(LandmarkPoint(*mapping.get(index, (0.5, 0.5))) for index in range(21))


if __name__ == "__main__":
    unittest.main()
