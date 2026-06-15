from __future__ import annotations

import logging
import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from airdeck.camera.frame_queue import FrameEnvelope, LatestFrameQueue


class FrameSink(Protocol):
    def connect(self) -> None: ...

    def publish(self, frame: Any, *, frame_index: int, timestamp_monotonic: float) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PublisherStats:
    frames_published: int
    frames_skipped: int
    fps: float
    last_frame_index: int | None


class NoopFrameSink:
    """Test/demo sink used when the LiveKit SDK is not configured."""

    def __init__(self) -> None:
        self.frames: list[int] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def publish(self, frame: Any, *, frame_index: int, timestamp_monotonic: float) -> None:
        _ = frame, timestamp_monotonic
        self.frames.append(frame_index)

    def close(self) -> None:
        self.connected = False


class LiveKitFrameSink:
    def __init__(self, *, url: str, token: str, track_name: str = "airdeck-camera") -> None:
        self._url = url
        self._token = token
        self._track_name = track_name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._room: Any | None = None
        self._source: Any | None = None
        self._track: Any | None = None
        self._width: int | None = None
        self._height: int | None = None
        self.connected = False

    def connect(self) -> None:
        try:
            from livekit import rtc  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install the 'overshoot' extra to publish with LiveKit") from exc

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._room = rtc.Room(loop=self._loop)
        self._loop.run_until_complete(self._room.connect(self._url, self._token))
        self.connected = True

    def publish(self, frame: Any, *, frame_index: int, timestamp_monotonic: float) -> None:
        _ = frame_index
        if not self.connected or self._loop is None or self._room is None:
            raise RuntimeError("LiveKit sink is not connected")
        rgb_frame = bgr_frame_to_rgb24(frame)
        height, width = rgb_frame.shape[:2]
        if self._source is None or self._width != width or self._height != height:
            self._create_track(width=width, height=height)

        from livekit import rtc  # type: ignore[import-not-found]

        video_frame = rtc.VideoFrame(
            width,
            height,
            rtc.VideoBufferType.RGB24,
            memoryview(rgb_frame).tobytes(),
        )
        timestamp_us = int(timestamp_monotonic * 1_000_000)
        self._source.capture_frame(video_frame, timestamp_us=timestamp_us)

    def close(self) -> None:
        if self._loop is None:
            return
        try:
            if self._room is not None:
                self._loop.run_until_complete(self._room.disconnect())
        finally:
            self._loop.close()
            self._loop = None
            self._room = None
            self._source = None
            self._track = None
            self.connected = False

    def _create_track(self, *, width: int, height: int) -> None:
        if self._room is None or self._loop is None:
            raise RuntimeError("LiveKit room is not connected")
        from livekit import rtc  # type: ignore[import-not-found]

        self._source = rtc.VideoSource(width, height)
        self._track = rtc.LocalVideoTrack.create_video_track(self._track_name, self._source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_CAMERA
        self._loop.run_until_complete(self._room.local_participant.publish_track(self._track, options))
        self._width = width
        self._height = height


class OvershootPublisher:
    def __init__(
        self,
        frame_queue: LatestFrameQueue,
        sink: FrameSink,
        *,
        target_fps: float = 18.0,
        resize: Callable[[Any], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        if target_fps <= 0 or target_fps > 20:
            raise ValueError("publisher target_fps must be in the range (0, 20]")
        self._frame_queue = frame_queue
        self._sink = sink
        self._target_fps = target_fps
        self._resize = resize or resize_frame_to_480p
        self._clock = clock
        self._sleeper = sleeper
        self._logger = logger or logging.getLogger("airdeck.overshoot.publisher")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_published = 0
        self._frames_skipped = 0
        self._last_frame_index: int | None = None
        self._started_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._started_at = self._clock()
        self._thread = threading.Thread(target=self._run, name="airdeck-overshoot-publisher", daemon=True)
        self._thread.start()
        self._logger.info("overshoot_publisher_started fps=%s", self._target_fps)

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = 2.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def stats(self) -> PublisherStats:
        elapsed = max((self._clock() - self._started_at) if self._started_at is not None else 0.0, 0.001)
        return PublisherStats(
            frames_published=self._frames_published,
            frames_skipped=self._frames_skipped,
            fps=self._frames_published / elapsed,
            last_frame_index=self._last_frame_index,
        )

    def publish_once(self) -> bool:
        envelope = self._frame_queue.peek_latest()
        if envelope is None:
            return False
        return self._publish_envelope(envelope)

    def _run(self) -> None:
        interval = 1.0 / self._target_fps
        try:
            self._sink.connect()
            while not self._stop_event.is_set():
                started = self._clock()
                self.publish_once()
                elapsed = self._clock() - started
                remaining = interval - elapsed
                if remaining > 0:
                    self._stop_event.wait(remaining)
        finally:
            self._sink.close()
            stats = self.stats()
            self._logger.info(
                "overshoot_publisher_stopped frames_published=%s fps=%.1f",
                stats.frames_published,
                stats.fps,
            )

    def _publish_envelope(self, envelope: FrameEnvelope) -> bool:
        if envelope.frame_index == self._last_frame_index:
            self._frames_skipped += 1
            return False
        frame = self._resize(envelope.frame)
        self._sink.publish(
            frame,
            frame_index=envelope.frame_index,
            timestamp_monotonic=envelope.timestamp_monotonic,
        )
        self._frames_published += 1
        self._last_frame_index = envelope.frame_index
        return True


def resize_frame_to_480p(frame: Any) -> Any:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2:
        return frame
    height, width = int(shape[0]), int(shape[1])
    long_edge = max(width, height)
    if long_edge <= 480:
        return frame
    scale = 480 / long_edge
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return frame
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))


def bgr_frame_to_rgb24(frame: Any) -> Any:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) != 3 or shape[2] < 3:
        raise ValueError("LiveKit publishing requires a color frame with at least three channels")
    return frame[:, :, 2::-1].copy()
