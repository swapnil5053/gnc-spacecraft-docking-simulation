from __future__ import annotations

import math

import numpy as np

from spacecraft_docking.math_utils import quat_from_euler
from spacecraft_docking.types import RigidBodyState, Scenario


def _state(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    euler_deg: tuple[float, float, float],
    angular_rate_deg_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidBodyState:
    roll, pitch, yaw = (math.radians(value) for value in euler_deg)
    return RigidBodyState(
        position=np.array(position, dtype=float),
        velocity=np.array(velocity, dtype=float),
        orientation=quat_from_euler(roll, pitch, yaw),
        angular_velocity=np.radians(np.array(angular_rate_deg_s, dtype=float)),
    )


SCENARIOS: dict[str, Scenario] = {
    # The station drifts slowly through the local frame and rotates gently,
    # so docking requires matching both linear and angular motion.
    "nominal": Scenario(
        name="nominal",
        description="Centered approach in microgravity with mild sensor and actuator noise.",
        initial_chaser_state=_state((6.0, 0.0, 0.0), (0.015, 0.006, -0.003), (0.0, 0.0, 180.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=180.0,
    ),
    "offset_start": Scenario(
        name="offset_start",
        description="Initial lateral offset and yaw error that should still converge.",
        initial_chaser_state=_state((6.0, 0.3, -0.18), (0.015, 0.006, -0.003), (0.0, 1.0, 186.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=220.0,
    ),
    "sensor_noise": Scenario(
        name="sensor_noise",
        description="Higher sensor noise with the same docking objective.",
        initial_chaser_state=_state((6.0, -0.12, 0.08), (0.015, 0.006, -0.003), (0.0, -1.0, 184.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=180.0,
        sensor_noise_scale=1.5,
    ),
    "actuator_saturation": Scenario(
        name="actuator_saturation",
        description="Reduced force and torque authority; docking should be slower but stable.",
        initial_chaser_state=_state((5.5, 0.25, 0.2), (0.015, 0.006, -0.003), (0.0, 0.0, 188.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=240.0,
        actuator_force_scale=0.55,
        actuator_torque_scale=0.7,
    ),
    "unsafe_closing_speed": Scenario(
        name="unsafe_closing_speed",
        description="Starts near the target with excessive closing speed and should abort.",
        initial_chaser_state=_state((1.25, 0.01, 0.0), (-0.25, 0.006, -0.003), (0.0, 0.0, 180.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=35.0,
    ),
    "large_misalignment": Scenario(
        name="large_misalignment",
        description="Starts too close with excessive angular misalignment and should retreat.",
        initial_chaser_state=_state((1.1, 0.28, -0.18), (-0.08, 0.0, 0.0), (0.0, 0.0, 145.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=40.0,
    ),
    "no_sensor_lock": Scenario(
        name="no_sensor_lock",
        description="Temporary target loss during the close approach should trigger retreat.",
        initial_chaser_state=_state((6.0, 0.12, -0.1), (0.015, 0.006, -0.003), (0.0, 0.0, 184.0)),
        initial_target_state=_state((0.0, -0.4, 0.15), (0.0, 0.006, -0.003), (0.0, 0.0, 0.0), (0.0, 0.02, 0.03)),
        duration=150.0,
        sensor_blackouts=[(80.0, 84.0)],
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario '{name}'. Available: {', '.join(sorted(SCENARIOS))}") from exc


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS)
