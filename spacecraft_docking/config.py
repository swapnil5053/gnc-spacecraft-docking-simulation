from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .math_utils import quat_from_axis_angle


@dataclass
class SimulationConfig:
    dt_physics: float = 0.01
    sensor_period: float = 0.05
    control_period: float = 0.05
    max_time: float = 80.0

    mass_kg: float = 120.0
    inertia_kgm2: np.ndarray = field(
        default_factory=lambda: np.array([45.0, 40.0, 30.0], dtype=float)
    )
    max_force_newton: float = 8.0
    max_torque_nm: float = 3.0

    position_kp: float = 1.5
    position_kd: float = 2.4
    attitude_kp: float = 7.5
    attitude_kd: float = 4.0

    position_noise_std_m: float = 0.01
    velocity_noise_std_mps: float = 0.015
    orientation_noise_std_deg: float = 0.5
    gyro_noise_std_deg_s: float = 0.2
    actuator_noise_std_force_n: float = 0.08
    actuator_noise_std_torque_nm: float = 0.02

    estimator_position_alpha: float = 0.55
    estimator_velocity_alpha: float = 0.45
    estimator_orientation_alpha: float = 0.45
    estimator_timeout_s: float = 0.4

    hold_point_x_m: float = 2.0
    retreat_point_x_m: float = 3.0
    final_approach_speed_mps: float = 0.04

    capture_radius_m: float = 0.10
    alignment_tolerance_m: float = 0.03
    angular_tolerance_deg: float = 5.0
    closing_speed_limit_mps: float = 0.05
    angular_rate_limit_deg_s: float = 2.0

    abort_lateral_limit_m: float = 0.12
    abort_angular_limit_deg: float = 12.0
    abort_closing_speed_mps: float = 0.12
    abort_sensor_timeout_s: float = 1.0

    target_docking_port_local: np.ndarray = field(
        default_factory=lambda: np.array([0.72, 0.0, 0.0], dtype=float)
    )
    chaser_docking_port_local: np.ndarray = field(
        default_factory=lambda: np.array([0.42, 0.0, 0.0], dtype=float)
    )
    desired_docking_orientation: np.ndarray = field(
        default_factory=lambda: quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
    )
