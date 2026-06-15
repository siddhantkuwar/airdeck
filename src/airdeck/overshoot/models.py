from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModelSelectionError(RuntimeError):
    """Raised when Overshoot returns no ready hosted model."""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    status: str
    owned_by: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


def parse_models(payload: dict[str, Any]) -> tuple[ModelInfo, ...]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelSelectionError("models response must contain a data list")

    models: list[ModelInfo] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        status = item.get("status")
        if isinstance(model_id, str) and isinstance(status, str):
            owned_by = item.get("owned_by")
            models.append(
                ModelInfo(
                    id=model_id,
                    status=status,
                    owned_by=owned_by if isinstance(owned_by, str) else None,
                )
            )
    return tuple(models)


def choose_ready_model(models: tuple[ModelInfo, ...]) -> ModelInfo:
    for model in models:
        if model.is_ready:
            return model
    raise ModelSelectionError("no ready Overshoot model available")
