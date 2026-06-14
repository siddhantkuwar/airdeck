from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from airdeck.camera.frame_queue import LatestFrameQueue


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class CaptureDevice(Protocol):
    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int], CaptureDevice]
ErrorCallback = Callable[[BaseException], None]


class OpenCVCapture:
    def __init__(self, camera_index: int) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise CameraError(
                "OpenCV is required for webcam capture. Install with: pip install -e '.[camera]'"
            ) from exc

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(camera_index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture.set(cv2.CAP_PROP_FPS, 30)
        if not self._capture.isOpened():
            raise CameraError(f"Unable to open camera index {camera_index}")

    def read(self) -> tuple[bool, Any]:
        return self._capture.read()

    def release(self) -> None:
        self._capture.release()


class CaptureProducer:
    def __init__(
        self,
        frame_queue: LatestFrameQueue,
        *,
        camera_index: int = 0,
        target_fps: float = 30.0,
        capture_factory: CaptureFactory | None = None,
        logger: logging.Logger | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")

        self._frame_queue = frame_queue
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._capture_factory = capture_factory or OpenCVCapture
        self._logger = logger or logging.getLogger("airdeck.camera.producer")
        self._on_error = on_error
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: CaptureDevice | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        capture = self._capture_factory(self._camera_index)
        self._capture = capture
        self._logger.info(
            "camera_producer_started index=%s fps=%s", self._camera_index, self._target_fps
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(capture,),
            name="airdeck-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = 2.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self, capture: CaptureDevice) -> None:
        try:
            self._capture_loop(capture)
        except BaseException as exc:  # noqa: BLE001 - callback must see camera startup/read errors.
            self._logger.error("camera_producer_error error=%s", exc, exc_info=True)
            if self._on_error:
                self._on_error(exc)
        finally:
            capture.release()
            self._capture = None
            metrics = self._frame_queue.metrics()
            self._logger.info(
                "camera_producer_stopped frames_received=%s frames_dropped=%s",
                metrics.frames_received,
                metrics.frames_dropped,
            )

    def _capture_loop(self, capture: CaptureDevice) -> None:
        interval = 1.0 / self._target_fps
        while not self._stop_event.is_set():
            loop_started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                raise CameraError("Camera read failed")
            self._frame_queue.put_latest(frame, timestamp_monotonic=loop_started)

            elapsed = time.monotonic() - loop_started
            remaining = interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)
