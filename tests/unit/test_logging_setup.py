from __future__ import annotations

import io
import json
import unittest

from airdeck.config import Settings
from airdeck.logging_setup import configure_logging


class LoggingSetupTests(unittest.TestCase):
    def test_redacts_api_key_and_tokens(self) -> None:
        stream = io.StringIO()
        settings = Settings(overshoot_api_key="ovs-secret")
        logger = configure_logging(settings, stream=stream)

        logger.info(
            "Authorization: Bearer token-123 key=%s publish_token=livekit-token",
            "ovs-secret",
        )

        output = stream.getvalue()
        self.assertNotIn("ovs-secret", output)
        self.assertNotIn("token-123", output)
        self.assertNotIn("livekit-token", output)
        self.assertIn("[REDACTED]", output)

        payload = json.loads(output)
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "airdeck")

    def test_preserves_numeric_format_args(self) -> None:
        stream = io.StringIO()
        logger = configure_logging(Settings(overshoot_api_key="ovs-secret"), stream=stream)

        logger.info("confidence=%.2f", 0.835)

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["message"], "confidence=0.83")


if __name__ == "__main__":
    unittest.main()
