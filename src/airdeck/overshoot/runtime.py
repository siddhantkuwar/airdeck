from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from airdeck.budget.counters import BudgetCounters
from airdeck.budget.limiter import BudgetLimiter, BudgetLimits
from airdeck.camera.frame_queue import LatestFrameQueue
from airdeck.config import Settings
from airdeck.overshoot.client import OvershootClient
from airdeck.overshoot.inference import InferenceResult, InferenceScheduler
from airdeck.overshoot.publisher import FrameSink, LiveKitFrameSink, OvershootPublisher
from airdeck.overshoot.stream import OvershootStreamManager, StreamSession


@dataclass(frozen=True)
class RuntimeStatus:
    connection_state: str
    stream_id: str | None
    model_id: str | None
    publisher_fps: float
    request_count: int


class OvershootRuntime:
    def __init__(
        self,
        settings: Settings,
        frame_queue: LatestFrameQueue,
        *,
        client: OvershootClient | None = None,
        sink_factory: Callable[[StreamSession], FrameSink] | None = None,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._frame_queue = frame_queue
        self._clock = clock
        self._logger = logger or logging.getLogger("airdeck.overshoot.runtime")
        self._client = client or OvershootClient(api_key=settings.overshoot_api_key)
        self._sink_factory = sink_factory or _livekit_sink_from_session
        self._stream_manager = OvershootStreamManager(
            self._client,
            keepalive_interval_seconds=settings.keepalive_interval_seconds,
            clock=clock,
            logger=self._logger,
        )
        self._counters = BudgetCounters(started_at_monotonic=clock())
        self._limiter = BudgetLimiter(
            BudgetLimits(
                max_completion_requests=settings.max_completion_requests,
                max_session_minutes=settings.max_session_minutes,
                max_requests_per_minute=settings.max_requests_per_minute,
                max_inference_hz=settings.max_inference_hz,
            )
        )
        self._scheduler = InferenceScheduler(
            self._client,
            self._counters,
            self._limiter,
            max_completion_requests=settings.max_completion_requests,
            max_inference_hz=settings.max_inference_hz,
            default_inference_hz=settings.default_inference_hz,
            clock=clock,
            logger=self._logger,
        )
        self._publisher: OvershootPublisher | None = None

    @property
    def counters(self) -> BudgetCounters:
        return self._counters

    @property
    def scheduler(self) -> InferenceScheduler:
        return self._scheduler

    @property
    def session(self) -> StreamSession | None:
        return self._stream_manager.session

    def start(self) -> StreamSession:
        session = self._stream_manager.start()
        sink = self._sink_factory(session)
        self._publisher = OvershootPublisher(
            self._frame_queue,
            sink,
            target_fps=self._settings.publisher_fps,
            logger=self._logger,
        )
        self._publisher.start()
        self._counters.start_stream(self._clock())
        return session

    def renew_if_due(self) -> None:
        self._stream_manager.renew_if_due()

    def infer_once(self) -> InferenceResult | None:
        session = self._stream_manager.session
        if session is None:
            return None
        return self._scheduler.try_infer(
            stream_id=session.stream.id,
            model_id=session.model.id,
            target_hz=self._settings.default_inference_hz,
        )

    def stop(self) -> None:
        self._scheduler.shutdown()
        if self._publisher is not None:
            self._publisher.stop()
            self._publisher.join(timeout=2.0)
            self._publisher = None
        self._counters.stop_stream(self._clock())
        self._stream_manager.stop()

    def status(self) -> RuntimeStatus:
        session = self._stream_manager.session
        publisher_stats = self._publisher.stats() if self._publisher is not None else None
        return RuntimeStatus(
            connection_state="connected" if session is not None else "offline",
            stream_id=session.stream.id if session is not None else None,
            model_id=session.model.id if session is not None else None,
            publisher_fps=publisher_stats.fps if publisher_stats is not None else 0.0,
            request_count=self._counters.total_completion_requests,
        )


def _livekit_sink_from_session(session: StreamSession) -> FrameSink:
    return LiveKitFrameSink(url=session.stream.publish.url, token=session.stream.publish.token)
