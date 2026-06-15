from __future__ import annotations

import time
import unittest
from typing import Any

from airdeck.camera.frame_queue import LatestFrameQueue
from airdeck.config import Settings
from airdeck.overshoot.publisher import NoopFrameSink
from airdeck.overshoot.runtime import OvershootRuntime
from airdeck.overshoot.stream import StreamSession


STREAM_PAYLOAD = {
    "id": "stream-1",
    "state": "active",
    "publish": {"type": "livekit", "url": "wss://room", "token": "token-1"},
    "expires_at_ms": 123,
    "ttl_seconds": 300,
}


class FakeClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.completions: list[dict[str, Any]] = []

    def list_models(self) -> dict[str, Any]:
        return {"data": [{"id": "model-ready", "status": "ready"}]}

    def create_stream(self) -> dict[str, Any]:
        return dict(STREAM_PAYLOAD)

    def keepalive_stream(self, stream_id: str) -> dict[str, Any]:
        payload = dict(STREAM_PAYLOAD)
        payload["id"] = stream_id
        return payload

    def delete_stream(self, stream_id: str) -> dict[str, Any]:
        self.deleted.append(stream_id)
        return {"id": stream_id, "deleted": True}

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.completions.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"gesture":"OPEN_PALM","confidence":0.92,'
                        '"hand_visible":true,"description":"Open hand"}'
                    }
                }
            ]
        }


class RuntimeTests(unittest.TestCase):
    def test_start_infer_and_stop_delete_stream(self) -> None:
        client = FakeClient()
        frame_queue = LatestFrameQueue()
        frame_queue.put_latest("frame", timestamp_monotonic=1.0)
        settings = Settings(overshoot_api_key="ovs-test", publisher_fps=18.0)
        sinks: list[NoopFrameSink] = []

        def sink_factory(_session: StreamSession) -> NoopFrameSink:
            sink = NoopFrameSink()
            sinks.append(sink)
            return sink

        runtime = OvershootRuntime(settings, frame_queue, client=client, sink_factory=sink_factory)
        session = runtime.start()
        time.sleep(0.05)
        result = runtime.infer_once()
        runtime.stop()

        self.assertEqual(session.stream.id, "stream-1")
        self.assertEqual(result.status, "OK")
        self.assertEqual(client.completions[0]["messages"][0]["content"][1]["image_url"]["url"], "ovs://streams/stream-1?frame_index=-1")
        self.assertEqual(client.deleted, ["stream-1"])
        self.assertFalse(sinks[0].connected)


if __name__ == "__main__":
    unittest.main()
