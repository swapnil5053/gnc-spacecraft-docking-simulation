from __future__ import annotations

import numpy as np

from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import clamp_norm, quat_error, quat_to_axis_angle, quat_to_rotmat
from spacecraft_docking.types import ControlCommand, Estimate, GuidanceSetpoint, RigidBodyState


class DockingController:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def compute(
        self,
        chaser_state: RigidBodyState,
        target_state: RigidBodyState,
        estimate: Estimate,
        setpoint: GuidanceSetpoint,
    ) -> ControlCommand:
        position_error_target = setpoint.desired_position - estimate.relative_position
        velocity_error_target = setpoint.desired_velocity - estimate.relative_velocity
        translational_kp = np.array([1.55, 2.35, 2.35], dtype=float)
        translational_kd = np.array([5.4, 4.2, 4.2], dtype=float)
        force_target = translational_kp * position_error_target + translational_kd * velocity_error_target

        target_rotation = quat_to_rotmat(target_state.orientation)
        chaser_rotation = quat_to_rotmat(chaser_state.orientation)
        force_world = target_rotation @ force_target
        force_body = chaser_rotation.T @ force_world

        error_quaternion = quat_error(estimate.relative_orientation, setpoint.desired_orientation)
        axis, angle = quat_to_axis_angle(error_quaternion)
        torque_body = self.config.attitude_kp * axis * angle - self.config.attitude_kd * estimate.relative_angular_velocity

        return ControlCommand(
            force_cmd_body=clamp_norm(force_body, self.config.max_force_newton),
            torque_cmd_body=clamp_norm(torque_body, self.config.max_torque_nm),
        )
