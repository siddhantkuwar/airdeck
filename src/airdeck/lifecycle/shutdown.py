from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass


ShutdownCallback = Callable[[], None]


@dataclass(frozen=True)
class ShutdownStep:
    name: str
    callback: ShutdownCallback
    priority: int
    sequence: int


class ShutdownManager:
    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("airdeck.lifecycle.shutdown")
        self._lock = threading.Lock()
        self._steps: list[ShutdownStep] = []
        self._next_sequence = 0
        self._has_shutdown = False

    def register(self, name: str, callback: ShutdownCallback, *, priority: int = 0) -> None:
        with self._lock:
            self._steps.append(
                ShutdownStep(
                    name=name,
                    callback=callback,
                    priority=priority,
                    sequence=self._next_sequence,
                )
            )
            self._next_sequence += 1

    def shutdown(self, reason: str = "requested") -> None:
        with self._lock:
            if self._has_shutdown:
                return
            self._has_shutdown = True
            steps = sorted(self._steps, key=lambda step: (step.priority, step.sequence), reverse=True)

        self._logger.info("shutdown_started reason=%s", reason)
        for step in steps:
            try:
                self._logger.info("shutdown_step_started name=%s", step.name)
                step.callback()
                self._logger.info("shutdown_step_completed name=%s", step.name)
            except BaseException as exc:  # noqa: BLE001 - shutdown should continue through failures.
                self._logger.error("shutdown_step_failed name=%s error=%s", step.name, exc, exc_info=True)
        self._logger.info("shutdown_completed reason=%s", reason)
