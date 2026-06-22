from __future__ import annotations

import unittest

from scenarios import get_scenario
from simulation import DockingSimulation
from spacecraft_docking.config import SimulationConfig


class DockingSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulation = DockingSimulation(SimulationConfig())

    def test_nominal_repeated_success_rate(self) -> None:
        successes = 0
        for seed in range(10):
            result = self.simulation.run(get_scenario("nominal"), seed=seed)
            successes += int(result.final_status.docked)
        self.assertGreaterEqual(successes, 9)

    def test_offset_start_docks(self) -> None:
        result = self.simulation.run(get_scenario("offset_start"), seed=11)
        self.assertTrue(result.final_status.docked)

    def test_sensor_noise_docks(self) -> None:
        result = self.simulation.run(get_scenario("sensor_noise"), seed=13)
        self.assertTrue(result.final_status.docked)

    def test_actuator_saturation_docks(self) -> None:
        result = self.simulation.run(get_scenario("actuator_saturation"), seed=17)
        self.assertTrue(result.final_status.docked)

    def test_unsafe_closing_speed_aborts(self) -> None:
        result = self.simulation.run(get_scenario("unsafe_closing_speed"), seed=3)
        self.assertTrue(result.final_status.aborted)
        self.assertFalse(result.final_status.docked)

    def test_large_misalignment_aborts(self) -> None:
        result = self.simulation.run(get_scenario("large_misalignment"), seed=5)
        self.assertTrue(result.final_status.aborted)
        self.assertFalse(result.final_status.docked)

    def test_sensor_blackout_aborts(self) -> None:
        result = self.simulation.run(get_scenario("no_sensor_lock"), seed=19)
        self.assertTrue(result.final_status.aborted)
        self.assertFalse(result.final_status.docked)


if __name__ == "__main__":
    unittest.main()
