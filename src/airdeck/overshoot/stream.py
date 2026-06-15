from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

from airdeck.overshoot.client import OvershootClient
from airdeck.overshoot.models import ModelInfo, choose_ready_model, parse_models


class StreamLifecycleError(RuntimeError):
    """Raised when Overshoot stream lifecycle payloads are invalid."""


@dataclass(frozen=True)
class PublishTarget:
    type: str
    url: str
    token: str


@dataclass(frozen=True)
class StreamInfo:
    id: str
    state: str
    publish: PublishTarget
    expires_at_ms: int
    ttl_seconds: int
    stream_time_ms: float = 0.0


@dataclass(frozen=True)
class StreamSession:
    model: ModelInfo
    stream: StreamInfo
    started_at_monotonic: float
    last_keepalive_monotonic: float


class OvershootStreamManager:
    def __init__(
        self,
        client: OvershootClient,
        *,
        keepalive_interval_seconds: float = 20.0,
        clock: callable = time.monotonic,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client
        self._keepalive_interval_seconds = keepalive_interval_seconds
        self._clock = clock
        self._logger = logger or logging.getLogger("airdeck.overshoot.stream")
        self._session: StreamSession | None = None

    @property
    def session(self) -> StreamSession | None:
        return self._session

    def start(self) -> StreamSession:
        if self._session is not None:
            return self._session
        model = choose_ready_model(parse_models(self._client.list_models()))
        stream = parse_stream_info(self._client.create_stream())
        now = self._clock()
        self._session = StreamSession(
            model=model,
            stream=stream,
            started_at_monotonic=now,
            last_keepalive_monotonic=now,
        )
        self._logger.info("overshoot_stream_created stream_id=%s model=%s", stream.id, model.id)
        return self._session

    def renew_if_due(self) -> StreamSession | None:
        if self._session is None:
            return None
        now = self._clock()
        if now - self._session.last_keepalive_monotonic < self._keepalive_interval_seconds:
            return self._session
        renewed = parse_stream_info(self._client.keepalive_stream(self._session.stream.id))
        self._session = replace(
            self._session,
            stream=renewed,
            last_keepalive_monotonic=now,
        )
        self._logger.info("overshoot_stream_keepalive stream_id=%s", renewed.id)
        return self._session

    def stop(self) -> None:
        if self._session is None:
            return
        stream_id = self._session.stream.id
        try:
            self._client.delete_stream(stream_id)
            self._logger.info("overshoot_stream_deleted stream_id=%s", stream_id)
        finally:
            self._session = None


def parse_stream_info(payload: dict[str, Any]) -> StreamInfo:
    stream_id = payload.get("id")
    state = payload.get("state", "active")
    publish = payload.get("publish")
    expires_at_ms = payload.get("expires_at_ms")
    ttl_seconds = payload.get("ttl_seconds")
    stream_time_ms = payload.get("stream_time_ms", 0.0)

    if not isinstance(stream_id, str) or not stream_id:
        raise StreamLifecycleError("stream payload missing id")
    if not isinstance(state, str):
        raise StreamLifecycleError("stream payload missing state")
    if not isinstance(publish, dict):
        raise StreamLifecycleError("stream payload missing publish target")
    target_type = publish.get("type")
    url = publish.get("url")
    token = publish.get("token")
    if not all(isinstance(value, str) and value for value in (target_type, url, token)):
        raise StreamLifecycleError("publish target missing type, url, or token")
    if not isinstance(expires_at_ms, int):
        raise StreamLifecycleError("stream payload missing expires_at_ms")
    if not isinstance(ttl_seconds, int):
        raise StreamLifecycleError("stream payload missing ttl_seconds")
    if not isinstance(stream_time_ms, int | float):
        raise StreamLifecycleError("stream_time_ms must be numeric")

    return StreamInfo(
        id=stream_id,
        state=state,
        publish=PublishTarget(type=target_type, url=url, token=token),
        expires_at_ms=expires_at_ms,
        ttl_seconds=ttl_seconds,
        stream_time_ms=float(stream_time_ms),
    )
