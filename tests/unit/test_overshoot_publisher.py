from __future__ import annotations

import unittest

import numpy as np

from airdeck.camera.frame_queue import LatestFrameQueue
from airdeck.overshoot.publisher import NoopFrameSink, OvershootPublisher, bgr_frame_to_rgb24


class PublisherTests(unittest.TestCase):
    def test_publish_once_uses_latest_frame_and_skips_duplicates(self) -> None:
        frame_queue = LatestFrameQueue()
        sink = NoopFrameSink()
        publisher = OvershootPublisher(frame_queue, sink)
        sink.connect()

        frame_queue.put_latest("first", timestamp_monotonic=1.0)
        self.assertTrue(publisher.publish_once())
        self.assertFalse(publisher.publish_once())
        frame_queue.put_latest("second", timestamp_monotonic=2.0)
        self.assertTrue(publisher.publish_once())

        self.assertEqual(sink.frames, [1, 2])
        self.assertEqual(publisher.stats().frames_skipped, 1)

    def test_rejects_publisher_rates_above_twenty_fps(self) -> None:
        with self.assertRaises(ValueError):
            OvershootPublisher(LatestFrameQueue(), NoopFrameSink(), target_fps=21)

    def test_bgr_frame_to_rgb24_converts_opencv_channel_order(self) -> None:
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)

        rgb = bgr_frame_to_rgb24(bgr)

        self.assertEqual(rgb.tolist(), [[[30, 20, 10]]])


if __name__ == "__main__":
    unittest.main()
