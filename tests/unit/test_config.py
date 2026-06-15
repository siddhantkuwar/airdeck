from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from airdeck.config import ConfigurationError, load_settings


class LoadSettingsTests(unittest.TestCase):
    def test_loads_api_key_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("OVERSHOOT_API_KEY=ovs-test\n", encoding="utf-8")

            settings = load_settings(env_file=env_file, environ={})

        self.assertEqual(settings.overshoot_api_key, "ovs-test")
        self.assertTrue(settings.demo_mode)
        self.assertEqual(settings.max_completion_requests, 80)
        self.assertEqual(settings.max_session_minutes, 5)
        self.assertEqual(settings.max_requests_per_minute, 60)
        self.assertEqual(settings.max_inference_hz, 2.0)
        self.assertEqual(settings.publisher_fps, 18.0)

    def test_environment_wins_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("OVERSHOOT_API_KEY=ovs-dotenv\n", encoding="utf-8")

            settings = load_settings(
                env_file=env_file,
                environ={"OVERSHOOT_API_KEY": "ovs-env", "AIRDECK_DEMO_MODE": "false"},
            )

        self.assertEqual(settings.overshoot_api_key, "ovs-env")
        self.assertFalse(settings.demo_mode)
        self.assertEqual(settings.max_completion_requests, 150)
        self.assertEqual(settings.max_session_minutes, 10)

    def test_missing_api_key_fails_startup(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_settings(env_file="/tmp/airdeck-missing-env", environ={})

    def test_invalid_positive_number_fails(self) -> None:
        with self.assertRaises(ConfigurationError):
            load_settings(
                env_file="/tmp/airdeck-missing-env",
                environ={"OVERSHOOT_API_KEY": "ovs-test", "AIRDECK_MAX_REQUESTS": "0"},
            )

    def test_loads_v2_limits_from_environment(self) -> None:
        settings = load_settings(
            env_file="/tmp/airdeck-missing-env",
            environ={
                "OVERSHOOT_API_KEY": "ovs-test",
                "AIRDECK_MAX_REQUESTS_PER_MINUTE": "30",
                "AIRDECK_MAX_INFERENCE_HZ": "1.5",
                "AIRDECK_PUBLISHER_FPS": "15",
                "AIRDECK_KEEPALIVE_SECONDS": "12",
            },
        )

        self.assertEqual(settings.max_requests_per_minute, 30)
        self.assertEqual(settings.max_inference_hz, 1.5)
        self.assertEqual(settings.publisher_fps, 15)
        self.assertEqual(settings.keepalive_interval_seconds, 12)


if __name__ == "__main__":
    unittest.main()
