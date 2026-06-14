from __future__ import annotations

import unittest

from airdeck.camera.gesture_feedback import BoundingBox, GestureFeedback
from airdeck.camera.preview import draw_feedback_overlay, frame_to_ppm, resize_frame_to_fit


class PreviewTests(unittest.TestCase):
    def test_frame_to_ppm_mirrors_bgr_frame(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        frame = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

        ppm = frame_to_ppm(frame, mirror=True)

        self.assertTrue(ppm.startswith(b"P6\n2 1\n255\n"))
        self.assertEqual(ppm.split(b"255\n", 1)[1], bytes([6, 5, 4, 3, 2, 1]))

    def test_draw_feedback_overlay_marks_bounding_box(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        feedback = GestureFeedback(
            gesture="HAND_TRACKED",
            confidence=0.62,
            hand_visible=True,
            motion_score=0.4,
            description="test",
            timestamp_monotonic=1.0,
            bounding_box=BoundingBox(x=10, y=12, width=20, height=18),
        )

        output = draw_feedback_overlay(frame, feedback)

        self.assertNotEqual(int(output.sum()), 0)

    def test_resize_frame_to_fit_scales_down(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        output = resize_frame_to_fit(frame, max_width=100, max_height=100)

        self.assertEqual(output.shape[:2], (50, 100))


if __name__ == "__main__":
    unittest.main()
