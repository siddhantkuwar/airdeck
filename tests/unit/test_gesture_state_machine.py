from __future__ import annotations

import unittest

from airdeck.gestures.state_machine import GestureStateMachine
from airdeck.overshoot.schemas import GestureInference


class GestureStateMachineTests(unittest.TestCase):
    def test_requires_repeated_confirmation_before_accepting(self) -> None:
        machine = GestureStateMachine()

        first = machine.observe(observation("OPEN_PALM"), now=0.0)
        second = machine.observe(observation("OPEN_PALM"), now=0.5)

        self.assertFalse(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(second.command.command, "TOGGLE_PLAYBACK")

    def test_held_gesture_executes_once_until_neutral(self) -> None:
        machine = GestureStateMachine()
        machine.observe(observation("OPEN_PALM"), now=0.0)
        self.assertTrue(machine.observe(observation("OPEN_PALM"), now=0.5).accepted)
        held = machine.observe(observation("OPEN_PALM"), now=3.0)
        machine.observe(observation("NO_GESTURE", confidence=0.0, hand_visible=False), now=3.5)
        machine.observe(observation("OPEN_PALM"), now=4.0)
        rearmed = machine.observe(observation("OPEN_PALM"), now=4.5)

        self.assertFalse(held.accepted)
        self.assertTrue(rearmed.accepted)

    def test_cooldown_blocks_duplicates(self) -> None:
        machine = GestureStateMachine()
        machine.observe(observation("THUMB_RIGHT"), now=0.0)
        self.assertTrue(machine.observe(observation("THUMB_RIGHT"), now=0.5).accepted)
        machine.observe(observation("NO_GESTURE", confidence=0.0, hand_visible=False), now=0.6)
        machine.observe(observation("THUMB_RIGHT"), now=0.7)
        blocked = machine.observe(observation("THUMB_RIGHT"), now=0.8)

        self.assertFalse(blocked.accepted)
        self.assertEqual(blocked.state, "COOLDOWN")

    def test_two_fingers_toggles_listening(self) -> None:
        machine = GestureStateMachine()
        machine.observe(observation("TWO_FINGERS"), now=0.0)
        decision = machine.observe(observation("TWO_FINGERS"), now=0.5)

        self.assertTrue(decision.accepted)
        self.assertFalse(decision.listening_enabled)


def observation(gesture: str, *, confidence: float = 0.9, hand_visible: bool = True) -> GestureInference:
    return GestureInference(
        gesture=gesture,
        confidence=confidence,
        hand_visible=hand_visible,
        description=gesture,
        epoch=1,
    )


if __name__ == "__main__":
    unittest.main()
