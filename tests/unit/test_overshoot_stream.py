from __future__ import annotations

import unittest
from typing import Any

from airdeck.overshoot.stream import OvershootStreamManager


STREAM_PAYLOAD = {
    "id": "3c90c3cc-0d44-4b50-8888-8dd25736052a",
    "state": "active",
    "publish": {"type": "livekit", "url": "wss://room", "token": "token-1"},
    "expires_at_ms": 123,
    "ttl_seconds": 300,
}


class FakeClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.keepalives = 0

    def list_models(self) -> dict[str, Any]:
        return {"data": [{"id": "model-ready", "status": "ready"}]}

    def create_stream(self) -> dict[str, Any]:
        return dict(STREAM_PAYLOAD)

    def keepalive_stream(self, stream_id: str) -> dict[str, Any]:
        self.keepalives += 1
        payload = dict(STREAM_PAYLOAD)
        payload["id"] = stream_id
        payload["publish"] = {"type": "livekit", "url": "wss://room", "token": "token-2"}
        return payload

    def delete_stream(self, stream_id: str) -> dict[str, Any]:
        self.deleted.append(stream_id)
        return {"id": stream_id, "deleted": True}


class StreamManagerTests(unittest.TestCase):
    def test_start_chooses_ready_model_and_creates_stream(self) -> None:
        manager = OvershootStreamManager(FakeClient(), clock=lambda: 10.0)

        session = manager.start()

        self.assertEqual(session.model.id, "model-ready")
        self.assertEqual(session.stream.publish.token, "token-1")

    def test_keepalive_refreshes_publish_token_when_due(self) -> None:
        now = 0.0
        client = FakeClient()
        manager = OvershootStreamManager(client, keepalive_interval_seconds=5.0, clock=lambda: now)
        manager.start()
        now = 6.0

        session = manager.renew_if_due()

        self.assertEqual(client.keepalives, 1)
        self.assertEqual(session.stream.publish.token, "token-2")

    def test_stop_deletes_stream(self) -> None:
        client = FakeClient()
        manager = OvershootStreamManager(client, clock=lambda: 0.0)
        session = manager.start()

        manager.stop()

        self.assertEqual(client.deleted, [session.stream.id])
        self.assertIsNone(manager.session)


if __name__ == "__main__":
    unittest.main()
