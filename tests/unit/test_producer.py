from __future__ import annotations

import time
import unittest
from typing import Any

from airdeck.camera.frame_queue import LatestFrameQueue
from airdeck.camera.producer import CaptureProducer


class FakeCapture:
    def __init__(self) -> None:
        self.read_count = 0
        self.release_count = 0

    def read(self) -> tuple[bool, Any]:
        self.read_count += 1
        return True, f"frame-{self.read_count}"

    def release(self) -> None:
        self.release_count += 1


class CaptureProducerTests(unittest.TestCase):
    def test_producer_never_grows_queue_beyond_one(self) -> None:
        frame_queue = LatestFrameQueue()
        fake_capture = FakeCapture()
        producer = CaptureProducer(
            frame_queue,
            target_fps=500.0,
            capture_factory=lambda _index: fake_capture,
        )

        producer.start()
        deadline = time.monotonic() + 1.0
        while frame_queue.metrics().frames_received < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        producer.stop()
        producer.join(timeout=1.0)

        self.assertFalse(producer.is_running)
        self.assertEqual(fake_capture.release_count, 1)
        self.assertLessEqual(frame_queue.qsize(), 1)
        self.assertGreaterEqual(frame_queue.metrics().frames_dropped, 1)

    def test_start_raises_when_capture_cannot_open(self) -> None:
        frame_queue = LatestFrameQueue()
        producer = CaptureProducer(
            frame_queue,
            capture_factory=lambda _index: (_ for _ in ()).throw(RuntimeError("no camera")),
        )

        with self.assertRaises(RuntimeError):
            producer.start()

        self.assertFalse(producer.is_running)


if __name__ == "__main__":
    unittest.main()
