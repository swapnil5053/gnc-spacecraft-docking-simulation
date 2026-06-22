from __future__ import annotations

from dataclasses import replace

import numpy as np

from control import DockingController
from estimation import RelativeStateEstimator
from guidance import AutoDockingGuidance
from physics import MicrogravityPhysics
from sensors import SensorModel
from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import angle_between_quaternions, quat_conjugate, quat_multiply, quat_to_rotmat
from spacecraft_docking.types import (
    ControlCommand,
    DockingStatus,
    Estimate,
    RigidBodyState,
    Scenario,
    SimulationResult,
)


class DockingSimulation:
    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.inertia_diag = np.diag(self.config.inertia_kgm2)

    def run(self, scenario: Scenario, seed: int = 7) -> SimulationResult:
        rng = np.random.default_rng(seed)
        physics = MicrogravityPhysics(self.config)
        sensors = SensorModel(self.config)
        estimator = RelativeStateEstimator(self.config)
        guidance = AutoDockingGuidance(self.config)
        controller = DockingController(self.config)

        chaser_state = self._clone_state(scenario.initial_chaser_state)
        target_state = self._clone_state(scenario.initial_target_state)
        command = ControlCommand(
            force_cmd_body=np.zeros(3, dtype=float),
            torque_cmd_body=np.zeros(3, dtype=float),
        )

        relative_position, relative_velocity, relative_orientation, relative_angular_velocity = self._true_relative_state(
            chaser_state,
            target_state,
        )
        estimate = Estimate(
            relative_position=relative_position.copy(),
            relative_velocity=relative_velocity.copy(),
            relative_orientation=relative_orientation.copy(),
            relative_angular_velocity=relative_angular_velocity.copy(),
            confidence=0.0,
            valid=False,
        )
        status = DockingStatus(
            phase=AutoDockingGuidance.ACQUIRE,
            docked=False,
            aborted=False,
            reason="Awaiting initialization",
            safety_flags={},
        )

        times: list[float] = []
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        angular_velocities: list[np.ndarray] = []
        target_positions: list[np.ndarray] = []
        target_velocities: list[np.ndarray] = []
        target_orientations: list[np.ndarray] = []
        target_angular_velocities: list[np.ndarray] = []
        relative_positions: list[np.ndarray] = []
        relative_velocities: list[np.ndarray] = []
        relative_angular_velocities: list[np.ndarray] = []
        estimated_positions: list[np.ndarray] = []
        estimated_velocities: list[np.ndarray] = []
        position_error_norm: list[float] = []
        lateral_error_norm: list[float] = []
        orientation_error_deg: list[float] = []
        closing_speed: list[float] = []
        command_forces_body: list[np.ndarray] = []
        command_torques_body: list[np.ndarray] = []
        chaser_speed: list[float] = []
        target_speed: list[float] = []
        chaser_angular_momentum_norm: list[float] = []
        target_angular_momentum_norm: list[float] = []
        phases: list[str] = []

        next_sensor_time = 0.0
        next_control_time = 0.0
        step_count = int(scenario.duration / self.config.dt_physics)

        for step in range(step_count + 1):
            timestamp = step * self.config.dt_physics
            packet = None

            if timestamp + 1e-9 >= next_sensor_time:
                packet = sensors.measure(
                    chaser_state=chaser_state,
                    target_state=target_state,
                    timestamp=timestamp,
                    sensor_available=scenario.sensor_available(timestamp),
                    rng=rng,
                    noise_scale=scenario.sensor_noise_scale,
                )
                next_sensor_time += self.config.sensor_period

            if packet is not None or timestamp + 1e-9 >= next_control_time:
                estimate = estimator.step(packet, timestamp)

            if timestamp + 1e-9 >= next_control_time:
                setpoint, status = guidance.update(estimate, timestamp)
                command = controller.compute(
                    chaser_state=chaser_state,
                    target_state=target_state,
                    estimate=estimate,
                    setpoint=setpoint,
                )
                next_control_time += self.config.control_period

            relative_position, relative_velocity, relative_orientation, relative_angular_velocity = self._true_relative_state(
                chaser_state,
                target_state,
            )

            self._log_sample(
                timestamp=timestamp,
                chaser_state=chaser_state,
                target_state=target_state,
                estimate=estimate,
                command=command,
                status=status,
                relative_position=relative_position,
                relative_velocity=relative_velocity,
                relative_orientation=relative_orientation,
                relative_angular_velocity=relative_angular_velocity,
                times=times,
                positions=positions,
                velocities=velocities,
                orientations=orientations,
                angular_velocities=angular_velocities,
                target_positions=target_positions,
                target_velocities=target_velocities,
                target_orientations=target_orientations,
                target_angular_velocities=target_angular_velocities,
                relative_positions=relative_positions,
                relative_velocities=relative_velocities,
                relative_angular_velocities=relative_angular_velocities,
                estimated_positions=estimated_positions,
                estimated_velocities=estimated_velocities,
                position_error_norm=position_error_norm,
                lateral_error_norm=lateral_error_norm,
                orientation_error_deg=orientation_error_deg,
                closing_speed=closing_speed,
                command_forces_body=command_forces_body,
                command_torques_body=command_torques_body,
                chaser_speed=chaser_speed,
                target_speed=target_speed,
                chaser_angular_momentum_norm=chaser_angular_momentum_norm,
                target_angular_momentum_norm=target_angular_momentum_norm,
                phases=phases,
            )

            if status.docked:
                break

            chaser_state = physics.step(
                state=chaser_state,
                command=command,
                dt=self.config.dt_physics,
                rng=rng,
                force_scale=scenario.actuator_force_scale,
                torque_scale=scenario.actuator_torque_scale,
            )
            target_state = physics.propagate_passive(target_state, self.config.dt_physics)

        metrics = self._summarize_metrics(
            times=np.asarray(times),
            position_error_norm=np.asarray(position_error_norm),
            orientation_error_deg=np.asarray(orientation_error_deg),
            closing_speed=np.asarray(closing_speed),
            status=status,
        )

        return SimulationResult(
            scenario_name=scenario.name,
            times=np.asarray(times),
            positions=np.asarray(positions),
            velocities=np.asarray(velocities),
            orientations=np.asarray(orientations),
            angular_velocities=np.asarray(angular_velocities),
            target_positions=np.asarray(target_positions),
            target_velocities=np.asarray(target_velocities),
            target_orientations=np.asarray(target_orientations),
            target_angular_velocities=np.asarray(target_angular_velocities),
            relative_positions=np.asarray(relative_positions),
            relative_velocities=np.asarray(relative_velocities),
            relative_angular_velocities=np.asarray(relative_angular_velocities),
            estimated_positions=np.asarray(estimated_positions),
            estimated_velocities=np.asarray(estimated_velocities),
            position_error_norm=np.asarray(position_error_norm),
            lateral_error_norm=np.asarray(lateral_error_norm),
            orientation_error_deg=np.asarray(orientation_error_deg),
            closing_speed=np.asarray(closing_speed),
            command_forces_body=np.asarray(command_forces_body),
            command_torques_body=np.asarray(command_torques_body),
            chaser_speed=np.asarray(chaser_speed),
            target_speed=np.asarray(target_speed),
            chaser_angular_momentum_norm=np.asarray(chaser_angular_momentum_norm),
            target_angular_momentum_norm=np.asarray(target_angular_momentum_norm),
            phases=phases,
            final_status=status,
            metrics=metrics,
        )

    def _clone_state(self, state: RigidBodyState) -> RigidBodyState:
        return replace(
            state,
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            orientation=state.orientation.copy(),
            angular_velocity=state.angular_velocity.copy(),
        )

    def _true_relative_state(
        self,
        chaser_state: RigidBodyState,
        target_state: RigidBodyState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        relative_orientation = quat_multiply(quat_conjugate(target_state.orientation), chaser_state.orientation)
        relative_angular_velocity_body = chaser_rotation.T @ (chaser_omega_world - target_omega_world)
        return (
            relative_position_target,
            relative_velocity_target,
            relative_orientation,
            relative_angular_velocity_body,
        )

    def _log_sample(
        self,
        timestamp: float,
        chaser_state: RigidBodyState,
        target_state: RigidBodyState,
        estimate: Estimate,
        command: ControlCommand,
        status: DockingStatus,
        relative_position: np.ndarray,
        relative_velocity: np.ndarray,
        relative_orientation: np.ndarray,
        relative_angular_velocity: np.ndarray,
        times: list[float],
        positions: list[np.ndarray],
        velocities: list[np.ndarray],
        orientations: list[np.ndarray],
        angular_velocities: list[np.ndarray],
        target_positions: list[np.ndarray],
        target_velocities: list[np.ndarray],
        target_orientations: list[np.ndarray],
        target_angular_velocities: list[np.ndarray],
        relative_positions: list[np.ndarray],
        relative_velocities: list[np.ndarray],
        relative_angular_velocities: list[np.ndarray],
        estimated_positions: list[np.ndarray],
        estimated_velocities: list[np.ndarray],
        position_error_norm: list[float],
        lateral_error_norm: list[float],
        orientation_error_deg: list[float],
        closing_speed: list[float],
        command_forces_body: list[np.ndarray],
        command_torques_body: list[np.ndarray],
        chaser_speed: list[float],
        target_speed: list[float],
        chaser_angular_momentum_norm: list[float],
        target_angular_momentum_norm: list[float],
        phases: list[str],
    ) -> None:
        times.append(timestamp)
        positions.append(chaser_state.position.copy())
        velocities.append(chaser_state.velocity.copy())
        orientations.append(chaser_state.orientation.copy())
        angular_velocities.append(chaser_state.angular_velocity.copy())
        target_positions.append(target_state.position.copy())
        target_velocities.append(target_state.velocity.copy())
        target_orientations.append(target_state.orientation.copy())
        target_angular_velocities.append(target_state.angular_velocity.copy())
        relative_positions.append(relative_position.copy())
        relative_velocities.append(relative_velocity.copy())
        relative_angular_velocities.append(relative_angular_velocity.copy())
        estimated_positions.append(estimate.relative_position.copy())
        estimated_velocities.append(estimate.relative_velocity.copy())
        position_error_norm.append(float(np.linalg.norm(relative_position)))
        lateral_error_norm.append(float(np.linalg.norm(relative_position[1:])))
        orientation_error_deg.append(
            float(np.degrees(angle_between_quaternions(relative_orientation, self.config.desired_docking_orientation)))
        )
        closing_speed.append(float(max(0.0, -relative_velocity[0])))
        command_forces_body.append(command.force_cmd_body.copy())
        command_torques_body.append(command.torque_cmd_body.copy())
        chaser_speed.append(float(np.linalg.norm(chaser_state.velocity)))
        target_speed.append(float(np.linalg.norm(target_state.velocity)))
        chaser_angular_momentum_norm.append(float(np.linalg.norm(self.inertia_diag @ chaser_state.angular_velocity)))
        target_angular_momentum_norm.append(float(np.linalg.norm(self.inertia_diag @ target_state.angular_velocity)))
        phases.append(status.phase)

    def _summarize_metrics(
        self,
        times: np.ndarray,
        position_error_norm: np.ndarray,
        orientation_error_deg: np.ndarray,
        closing_speed: np.ndarray,
        status: DockingStatus,
    ) -> dict[str, float]:
        metrics = {
            "final_distance_m": float(position_error_norm[-1]),
            "final_orientation_error_deg": float(orientation_error_deg[-1]),
            "max_closing_speed_mps": float(np.max(closing_speed)),
            "duration_s": float(times[-1]),
            "docked": float(status.docked),
            "aborted": float(status.aborted),
        }
        if status.docked:
            metrics["time_to_dock_s"] = float(times[-1])
        return metrics
