from __future__ import annotations

import unittest

from airdeck.ui.app import clamp01, drop_pressure, fps_ratio


class TelemetryScalingTests(unittest.TestCase):
    def test_fps_ratio_clamps_to_chart_range(self) -> None:
        self.assertEqual(fps_ratio(0, 30), 0.0)
        self.assertEqual(fps_ratio(15, 30), 0.5)
        self.assertEqual(fps_ratio(60, 30), 1.0)

    def test_drop_pressure_treats_drops_as_bottom_band(self) -> None:
        self.assertEqual(drop_pressure(0, 30), 0.0)
        self.assertLess(drop_pressure(30, 30), 0.35)
        self.assertEqual(drop_pressure(200, 30), 1.0)

    def test_clamp01_bounds_values(self) -> None:
        self.assertEqual(clamp01(-1), 0.0)
        self.assertEqual(clamp01(0.4), 0.4)
        self.assertEqual(clamp01(2), 1.0)


if __name__ == "__main__":
    unittest.main()
