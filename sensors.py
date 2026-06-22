from __future__ import annotations

import numpy as np

from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import (
    normalize,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
)
from spacecraft_docking.types import RigidBodyState, SensorPacket


class SensorModel:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def measure(
        self,
        chaser_state: RigidBodyState,
        target_state: RigidBodyState,
        timestamp: float,
        sensor_available: bool,
        rng: np.random.Generator,
        noise_scale: float = 1.0,
    ) -> SensorPacket:
        if not sensor_available:
            return SensorPacket(
                relative_position=np.zeros(3, dtype=float),
                relative_velocity=np.zeros(3, dtype=float),
                relative_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                imu_rates=np.zeros(3, dtype=float),
                timestamp=timestamp,
                valid=False,
            )

        chaser_rotation = quat_to_rotmat(chaser_state.orientation)
        target_rotation = quat_to_rotmat(target_state.orientation)
        chaser_omega_world = chaser_rotation @ chaser_state.angular_velocity
        target_omega_world = target_rotation @ target_state.angular_velocity

        chaser_port_world = chaser_state.position + chaser_rotation @ self.config.chaser_docking_port_local
        target_port_world = target_state.position + target_rotation @ self.config.target_docking_port_local

        chaser_port_velocity = (
            chaser_state.velocity
            + np.cross(chaser_omega_world, chaser_rotation @ self.config.chaser_docking_port_local)
        )
        target_port_velocity = (
            target_state.velocity
            + np.cross(target_omega_world, target_rotation @ self.config.target_docking_port_local)
        )

        relative_position_target = target_rotation.T @ (chaser_port_world - target_port_world)
        relative_velocity_target = target_rotation.T @ (chaser_port_velocity - target_port_velocity)

        noisy_position = relative_position_target + rng.normal(
            0.0,
            self.config.position_noise_std_m * noise_scale,
            size=3,
        )

        noise_axis = normalize(rng.normal(size=3))
        noise_angle = np.deg2rad(
            rng.normal(0.0, self.config.orientation_noise_std_deg * noise_scale)
        )
        orientation_noise = quat_from_axis_angle(noise_axis, noise_angle)
        noisy_orientation = quat_normalize(
            quat_multiply(
                quat_multiply(quat_conjugate(target_state.orientation), chaser_state.orientation),
                orientation_noise,
            )
        )

        noisy_velocity = relative_velocity_target + rng.normal(
            0.0,
            self.config.velocity_noise_std_mps * noise_scale,
            size=3,
        )

        relative_angular_rate_body = chaser_rotation.T @ (
            chaser_omega_world - target_omega_world
        )
        imu_rates = relative_angular_rate_body + np.deg2rad(
            rng.normal(
                0.0,
                self.config.gyro_noise_std_deg_s * noise_scale,
                size=3,
            )
        )

        return SensorPacket(
            relative_position=noisy_position,
            relative_velocity=noisy_velocity,
            relative_orientation=noisy_orientation,
            imu_rates=imu_rates,
            timestamp=timestamp,
            valid=True,
        )
