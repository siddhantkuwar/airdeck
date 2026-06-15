from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when local configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    overshoot_api_key: str
    demo_mode: bool = True
    max_completion_requests: int = 80
    max_session_minutes: int = 5
    max_requests_per_minute: int = 60
    max_inference_hz: float = 2.0
    default_inference_hz: float = 0.75
    publisher_fps: float = 18.0
    keepalive_interval_seconds: float = 20.0
    camera_index: int = 0
    target_camera_fps: float = 30.0
    log_level: str = "INFO"

    @property
    def redaction_secrets(self) -> tuple[str, ...]:
        return (self.overshoot_api_key,) if self.overshoot_api_key else ()


def load_settings(
    *,
    env_file: str | Path = ".env",
    require_api_key: bool = True,
    environ: dict[str, str] | None = None,
) -> Settings:
    values = dict(os.environ if environ is None else environ)
    _load_dotenv(Path(env_file), values)

    api_key = values.get("OVERSHOOT_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ConfigurationError("OVERSHOOT_API_KEY is required")

    demo_mode = _bool_value(values.get("AIRDECK_DEMO_MODE", "true"))
    default_max_requests = 80 if demo_mode else 150
    default_session_minutes = 5 if demo_mode else 10

    return Settings(
        overshoot_api_key=api_key,
        demo_mode=demo_mode,
        max_completion_requests=_int_value(
            values.get("AIRDECK_MAX_REQUESTS"), default_max_requests, "AIRDECK_MAX_REQUESTS"
        ),
        max_session_minutes=_int_value(
            values.get("AIRDECK_MAX_SESSION_MINUTES"),
            default_session_minutes,
            "AIRDECK_MAX_SESSION_MINUTES",
        ),
        max_requests_per_minute=_int_value(
            values.get("AIRDECK_MAX_REQUESTS_PER_MINUTE"),
            60,
            "AIRDECK_MAX_REQUESTS_PER_MINUTE",
        ),
        max_inference_hz=_float_value(
            values.get("AIRDECK_MAX_INFERENCE_HZ"), 2.0, "AIRDECK_MAX_INFERENCE_HZ"
        ),
        default_inference_hz=_float_value(
            values.get("AIRDECK_INFERENCE_HZ"), 0.75, "AIRDECK_INFERENCE_HZ"
        ),
        publisher_fps=_float_value(values.get("AIRDECK_PUBLISHER_FPS"), 18.0, "AIRDECK_PUBLISHER_FPS"),
        keepalive_interval_seconds=_float_value(
            values.get("AIRDECK_KEEPALIVE_SECONDS"),
            20.0,
            "AIRDECK_KEEPALIVE_SECONDS",
        ),
        camera_index=_int_value(values.get("AIRDECK_CAMERA_INDEX"), 0, "AIRDECK_CAMERA_INDEX"),
        target_camera_fps=_float_value(
            values.get("AIRDECK_CAMERA_FPS"), 30.0, "AIRDECK_CAMERA_FPS"
        ),
        log_level=values.get("AIRDECK_LOG_LEVEL", "INFO").upper(),
    )


def _load_dotenv(path: Path, values: dict[str, str]) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in values:
            continue
        values[key] = _strip_quotes(value.strip())


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value!r}")


def _int_value(value: str | None, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _float_value(value: str | None, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed
