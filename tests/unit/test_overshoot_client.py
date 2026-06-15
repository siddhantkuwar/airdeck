from __future__ import annotations

import unittest
from typing import Any

from airdeck.overshoot.client import HTTPResponse, OvershootClient
from airdeck.overshoot.models import choose_ready_model, parse_models


class FakeTransport:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, str], dict[str, Any] | None]] = []
        self.response = response or {"ok": True}

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> HTTPResponse:
        _ = timeout
        self.calls.append((method, url, headers, payload))
        return HTTPResponse(status=200, payload=self.response)


class OvershootClientTests(unittest.TestCase):
    def test_models_request_omits_authorization(self) -> None:
        transport = FakeTransport({"object": "list", "data": []})
        client = OvershootClient(api_key="ovs-secret", transport=transport)

        client.list_models()

        self.assertNotIn("Authorization", transport.calls[0][2])

    def test_authenticated_requests_use_bearer_header(self) -> None:
        transport = FakeTransport({"id": "stream-1", "deleted": True})
        client = OvershootClient(api_key="ovs-secret", transport=transport)

        client.delete_stream("stream-1")

        self.assertEqual(transport.calls[0][2]["Authorization"], "Bearer ovs-secret")

    def test_choose_ready_model(self) -> None:
        models = parse_models(
            {
                "data": [
                    {"id": "loading", "status": "loading"},
                    {"id": "ready-model", "status": "ready", "owned_by": "overshoot"},
                ]
            }
        )

        self.assertEqual(choose_ready_model(models).id, "ready-model")


if __name__ == "__main__":
    unittest.main()
