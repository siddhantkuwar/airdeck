from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Iterable, TextIO

from airdeck.config import Settings


_BEARER_RE = re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?bearer\s+)[^\s,'\"}]+")
_TOKEN_RE = re.compile(r"(?i)\b((?:publish|livekit|access)?_?token['\"]?\s*[:=]\s*['\"]?)[^,'\"}\s]+")


def configure_logging(
    settings: Settings | None = None,
    *,
    stream: TextIO | None = None,
    level: str | int | None = None,
) -> logging.Logger:
    logger = logging.getLogger("airdeck")
    logger.setLevel(level or (settings.log_level if settings else "INFO"))
    logger.propagate = False
    logger.handlers.clear()
    logger.filters.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter(settings.redaction_secrets if settings else ()))
    logger.addHandler(handler)
    return logger


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(str(record.msg), self._secrets)
        if record.args:
            record.args = tuple(
                redact_text(arg, self._secrets) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(event, sort_keys=True)


def redact_text(value: str, secrets: Iterable[str] = ()) -> str:
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", redacted)
    redacted = _TOKEN_RE.sub(r"\1[REDACTED]", redacted)
    return redacted
