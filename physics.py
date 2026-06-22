from __future__ import annotations

import numpy as np

from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import clamp_norm, integrate_quaternion, quat_to_rotmat
from spacecraft_docking.types import ControlCommand, RigidBodyState


class MicrogravityPhysics:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.inertia_matrix = np.diag(config.inertia_kgm2)
        self.inertia_inv = np.diag(1.0 / config.inertia_kgm2)

    def step(
        self,
        state: RigidBodyState,
        command: ControlCommand,
        dt: float,
        rng: np.random.Generator,
        force_scale: float = 1.0,
        torque_scale: float = 1.0,
    ) -> RigidBodyState:
        force_body = clamp_norm(
            command.force_cmd_body,
            self.config.max_force_newton * force_scale,
        )
        torque_body = clamp_norm(
            command.torque_cmd_body,
            self.config.max_torque_nm * torque_scale,
        )

        force_body = force_body + rng.normal(
            0.0,
            self.config.actuator_noise_std_force_n * force_scale,
            size=3,
        )
        torque_body = torque_body + rng.normal(
            0.0,
            self.config.actuator_noise_std_torque_nm * torque_scale,
            size=3,
        )

        rotation_matrix = quat_to_rotmat(state.orientation)
        acceleration_world = rotation_matrix @ force_body / self.config.mass_kg

        angular_momentum = self.inertia_matrix @ state.angular_velocity
        coriolis = np.cross(state.angular_velocity, angular_momentum)
        angular_acceleration = self.inertia_inv @ (torque_body - coriolis)

        next_velocity = state.velocity + acceleration_world * dt
        next_position = state.position + next_velocity * dt
        next_angular_velocity = state.angular_velocity + angular_acceleration * dt
        next_orientation = integrate_quaternion(
            state.orientation,
            next_angular_velocity,
            dt,
        )

        return RigidBodyState(
            position=next_position,
            velocity=next_velocity,
            orientation=next_orientation,
            angular_velocity=next_angular_velocity,
        )

    def propagate_passive(self, state: RigidBodyState, dt: float) -> RigidBodyState:
        next_position = state.position + state.velocity * dt
        next_orientation = integrate_quaternion(
            state.orientation,
            state.angular_velocity,
            dt,
        )
        return RigidBodyState(
            position=next_position,
            velocity=state.velocity.copy(),
            orientation=next_orientation,
            angular_velocity=state.angular_velocity.copy(),
        )
