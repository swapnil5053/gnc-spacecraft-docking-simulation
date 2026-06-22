from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RigidBodyState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray


@dataclass
class ControlCommand:
    force_cmd_body: np.ndarray
    torque_cmd_body: np.ndarray


@dataclass
class SensorPacket:
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    relative_orientation: np.ndarray
    imu_rates: np.ndarray
    timestamp: float
    valid: bool


@dataclass
class Estimate:
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    relative_orientation: np.ndarray
    relative_angular_velocity: np.ndarray
    confidence: float
    valid: bool


@dataclass
class GuidanceSetpoint:
    desired_position: np.ndarray
    desired_velocity: np.ndarray
    desired_orientation: np.ndarray


@dataclass
class DockingStatus:
    phase: str
    docked: bool
    aborted: bool
    reason: str
    safety_flags: dict[str, bool] = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    description: str
    initial_chaser_state: RigidBodyState
    initial_target_state: RigidBodyState
    duration: float
    sensor_blackouts: list[tuple[float, float]] = field(default_factory=list)
    sensor_noise_scale: float = 1.0
    actuator_force_scale: float = 1.0
    actuator_torque_scale: float = 1.0

    def sensor_available(self, timestamp: float) -> bool:
        return not any(start <= timestamp <= end for start, end in self.sensor_blackouts)


@dataclass
class SimulationResult:
    scenario_name: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    orientations: np.ndarray
    angular_velocities: np.ndarray
    target_positions: np.ndarray
    target_velocities: np.ndarray
    target_orientations: np.ndarray
    target_angular_velocities: np.ndarray
    relative_positions: np.ndarray
    relative_velocities: np.ndarray
    relative_angular_velocities: np.ndarray
    estimated_positions: np.ndarray
    estimated_velocities: np.ndarray
    position_error_norm: np.ndarray
    lateral_error_norm: np.ndarray
    orientation_error_deg: np.ndarray
    closing_speed: np.ndarray
    command_forces_body: np.ndarray
    command_torques_body: np.ndarray
    chaser_speed: np.ndarray
    target_speed: np.ndarray
    chaser_angular_momentum_norm: np.ndarray
    target_angular_momentum_norm: np.ndarray
    phases: list[str]
    final_status: DockingStatus
    metrics: dict[str, float]
