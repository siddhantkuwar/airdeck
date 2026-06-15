from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class OvershootAPIError(RuntimeError):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in {429, 503, None}

    @property
    def fatal(self) -> bool:
        return self.status in {401, 402, 403}


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    payload: dict[str, Any]


class HTTPTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> HTTPResponse: ...


class UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> HTTPResponse:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                response_body = response.read().decode("utf-8")
                parsed = json.loads(response_body) if response_body else {}
                if not isinstance(parsed, dict):
                    raise OvershootAPIError(response.status, "Overshoot response must be a JSON object")
                return HTTPResponse(status=response.status, payload=parsed)
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8")
            except OSError:
                error_body = ""
            message = _safe_error_message(error_body) or f"Overshoot HTTP {exc.code}"
            raise OvershootAPIError(exc.code, message) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OvershootAPIError(None, "Overshoot network request failed") from exc


class OvershootClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.overshoot.ai/v1",
        timeout_seconds: float = 10.0,
        transport: HTTPTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._transport = transport or UrllibTransport()

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/models", auth=False)

    def create_stream(self) -> dict[str, Any]:
        return self._request("POST", "/streams", auth=True)

    def get_stream(self, stream_id: str) -> dict[str, Any]:
        return self._request("GET", f"/streams/{stream_id}", auth=True)

    def keepalive_stream(self, stream_id: str) -> dict[str, Any]:
        return self._request("POST", f"/streams/{stream_id}/keepalive", auth=True)

    def delete_stream(self, stream_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/streams/{stream_id}", auth=True)

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/chat/completions", auth=True, payload=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth: bool,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = self._transport.request(
            method,
            f"{self._base_url}{path}",
            headers=headers,
            payload=payload,
            timeout=self._timeout_seconds,
        )
        if response.status >= 400:
            raise OvershootAPIError(response.status, f"Overshoot HTTP {response.status}")
        return response.payload


def _safe_error_message(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])
    if isinstance(payload.get("message"), str):
        return str(payload["message"])
    return ""
