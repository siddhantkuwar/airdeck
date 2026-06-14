from __future__ import annotations

import unittest

from airdeck.camera.frame_queue import LatestFrameQueue


class LatestFrameQueueTests(unittest.TestCase):
    def test_drops_stale_frame_when_full(self) -> None:
        frame_queue = LatestFrameQueue()

        first = frame_queue.put_latest("first", timestamp_monotonic=1.0)
        second = frame_queue.put_latest("second", timestamp_monotonic=2.0)

        self.assertEqual(frame_queue.maxsize, 1)
        self.assertEqual(frame_queue.qsize(), 1)
        self.assertEqual(first.frame_index, 1)
        self.assertEqual(second.frame_index, 2)
        self.assertEqual(frame_queue.peek_latest(), second)

        queued = frame_queue.get_latest()
        self.assertEqual(queued.frame, "second")

        metrics = frame_queue.metrics()
        self.assertEqual(metrics.frames_received, 2)
        self.assertEqual(metrics.frames_dropped, 1)
        self.assertEqual(metrics.latest_frame_index, 2)


if __name__ == "__main__":
    unittest.main()
