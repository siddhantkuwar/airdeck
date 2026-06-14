from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameEnvelope:
    frame: Any
    timestamp_monotonic: float
    frame_index: int


@dataclass(frozen=True)
class QueueMetrics:
    frames_received: int
    frames_dropped: int
    latest_frame_index: int | None
    qsize: int


class LatestFrameQueue:
    """Thread-safe maxsize=1 queue that drops stale frames before inserting fresh frames."""

    def __init__(self) -> None:
        self._queue: queue.Queue[FrameEnvelope] = queue.Queue(maxsize=1)
        self._metrics_lock = threading.Lock()
        self._latest: FrameEnvelope | None = None
        self._frames_received = 0
        self._frames_dropped = 0

    def put_latest(self, frame: Any, *, timestamp_monotonic: float | None = None) -> FrameEnvelope:
        timestamp = time.monotonic() if timestamp_monotonic is None else timestamp_monotonic
        with self._metrics_lock:
            self._frames_received += 1
            envelope = FrameEnvelope(
                frame=frame,
                timestamp_monotonic=timestamp,
                frame_index=self._frames_received,
            )
            self._latest = envelope

        try:
            self._queue.put_nowait(envelope)
        except queue.Full:
            try:
                self._queue.get_nowait()
                with self._metrics_lock:
                    self._frames_dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(envelope)
            except queue.Full:
                with self._metrics_lock:
                    self._frames_dropped += 1
        return envelope

    def get_latest(self, *, timeout: float | None = None) -> FrameEnvelope:
        if timeout is None:
            return self._queue.get_nowait()
        return self._queue.get(timeout=timeout)

    def peek_latest(self) -> FrameEnvelope | None:
        with self._metrics_lock:
            return self._latest

    def metrics(self) -> QueueMetrics:
        with self._metrics_lock:
            latest_index = self._latest.frame_index if self._latest else None
            return QueueMetrics(
                frames_received=self._frames_received,
                frames_dropped=self._frames_dropped,
                latest_frame_index=latest_index,
                qsize=self._queue.qsize(),
            )

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def qsize(self) -> int:
        return self._queue.qsize()
