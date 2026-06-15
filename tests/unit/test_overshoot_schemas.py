from __future__ import annotations

import unittest

from airdeck.overshoot.schemas import GestureParseError, parse_gesture_content


class GestureSchemaTests(unittest.TestCase):
    def test_accepts_strict_gesture_json(self) -> None:
        inference = parse_gesture_content(
            '{"gesture":"OPEN_PALM","confidence":0.93,"hand_visible":true,'
            '"description":"Open hand"}',
            response_epoch=2,
            current_epoch=2,
            latency_ms=42.0,
        )

        self.assertEqual(inference.gesture, "OPEN_PALM")
        self.assertEqual(inference.confidence, 0.93)
        self.assertEqual(inference.latency_ms, 42.0)

    def test_rejects_prose_wrapped_json(self) -> None:
        with self.assertRaises(GestureParseError):
            parse_gesture_content(
                'Here: {"gesture":"OPEN_PALM","confidence":0.93,'
                '"hand_visible":true,"description":"Open hand"}',
                response_epoch=1,
                current_epoch=1,
            )

    def test_rejects_unknown_gesture_and_bad_confidence(self) -> None:
        with self.assertRaises(GestureParseError):
            parse_gesture_content(
                '{"gesture":"WAVE","confidence":0.93,"hand_visible":true,'
                '"description":"Wave"}',
                response_epoch=1,
                current_epoch=1,
            )
        with self.assertRaises(GestureParseError):
            parse_gesture_content(
                '{"gesture":"OPEN_PALM","confidence":1.3,"hand_visible":true,'
                '"description":"Open hand"}',
                response_epoch=1,
                current_epoch=1,
            )

    def test_rejects_stale_epoch(self) -> None:
        with self.assertRaises(GestureParseError):
            parse_gesture_content(
                '{"gesture":"OPEN_PALM","confidence":0.9,"hand_visible":true,'
                '"description":"Open hand"}',
                response_epoch=1,
                current_epoch=2,
            )


if __name__ == "__main__":
    unittest.main()
