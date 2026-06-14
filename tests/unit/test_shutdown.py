from __future__ import annotations

import unittest

from airdeck.lifecycle.shutdown import ShutdownManager


class ShutdownManagerTests(unittest.TestCase):
    def test_shutdown_runs_once_in_priority_order(self) -> None:
        events: list[str] = []
        manager = ShutdownManager()
        manager.register("low", lambda: events.append("low"), priority=0)
        manager.register("high", lambda: events.append("high"), priority=10)

        manager.shutdown("test")
        manager.shutdown("second-call")

        self.assertEqual(events, ["high", "low"])

    def test_shutdown_continues_after_step_failure(self) -> None:
        events: list[str] = []
        manager = ShutdownManager()

        def fail() -> None:
            events.append("fail")
            raise RuntimeError("boom")

        manager.register("after", lambda: events.append("after"), priority=0)
        manager.register("fail", fail, priority=10)

        manager.shutdown("test")

        self.assertEqual(events, ["fail", "after"])


if __name__ == "__main__":
    unittest.main()
