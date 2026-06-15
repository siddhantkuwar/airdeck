from __future__ import annotations

import unittest

from airdeck.camera.frame_queue import LatestFrameQueue
from airdeck.overshoot.publisher import NoopFrameSink, OvershootPublisher


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


if __name__ == "__main__":
    unittest.main()
