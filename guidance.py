from __future__ import annotations

import math

import numpy as np

from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import angle_between_quaternions
from spacecraft_docking.types import DockingStatus, Estimate, GuidanceSetpoint


class AutoDockingGuidance:
    ACQUIRE = "Acquire Target"
    FAR_RANGE = "Far-Range Approach"
    ALIGN = "Axis Alignment Hold"
    FINAL = "Final Approach"
    CAPTURE = "Soft Capture"
    ABORT = "Abort / Retreat"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.phase = self.ACQUIRE
        self.reason = "Awaiting sensor lock"
        self.last_valid_time: float | None = None
        self.align_ready_since: float | None = None
        self.violation_since: float | None = None
        self.capture_ready_since: float | None = None
        self.final_target_x: float | None = None

    def update(self, estimate: Estimate, timestamp: float) -> tuple[GuidanceSetpoint, DockingStatus]:
        if estimate.valid:
            self.last_valid_time = timestamp

        metrics = self._compute_metrics(estimate)
        safety_flags = self._build_safety_flags(metrics, estimate, timestamp)

        if self.phase == self.ACQUIRE and estimate.valid:
            self.phase = self.FAR_RANGE
            self.reason = "Sensor lock acquired"

        if self.phase == self.FAR_RANGE:
            if self._close_range_violation(metrics):
                self.phase = self.ABORT
                self.reason = "Unsafe state detected near docking corridor"
            elif (
                metrics["axial_distance"] <= self.config.hold_point_x_m + 0.15
                and metrics["speed_norm"] <= 0.20
            ):
                self.phase = self.ALIGN
                self.reason = "Reached hold point"

        if self.phase == self.ALIGN:
            if safety_flags["sensor_timeout"]:
                self.phase = self.ABORT
                self.reason = "Lost target during alignment"
            elif (
                metrics["lateral_error"] <= 0.025
                and metrics["orientation_error_deg"] <= 5.0
                and metrics["speed_norm"] <= 0.10
            ):
                if self.align_ready_since is None:
                    self.align_ready_since = timestamp
                elif timestamp - self.align_ready_since >= 0.8:
                    self.phase = self.FINAL
                    self.reason = "Alignment constraints satisfied"
            else:
                self.align_ready_since = None

        if self.phase == self.FINAL:
            if (
                safety_flags["sensor_timeout"]
                or safety_flags["angular_limit"]
                or safety_flags["closing_speed_limit"]
            ):
                if self.violation_since is None:
                    self.violation_since = timestamp
                elif timestamp - self.violation_since >= 0.25:
                    self.phase = self.ABORT
                    self.reason = "Safety limit violated during final approach"
            else:
                self.violation_since = None

            if self.phase == self.FINAL and self._is_docked(metrics):
                if self.capture_ready_since is None:
                    self.capture_ready_since = timestamp
                elif timestamp - self.capture_ready_since >= 0.20:
                    self.phase = self.CAPTURE
                    self.reason = "Docking envelope satisfied"
            else:
                self.capture_ready_since = None

        if self.phase == self.ABORT and metrics["axial_distance"] >= self.config.retreat_point_x_m - 0.1:
            self.reason = "Retreated to safe hold point"

        setpoint = self._build_setpoint(estimate, metrics)
        status = DockingStatus(
            phase=self.phase,
            docked=self.phase == self.CAPTURE,
            aborted=self.phase == self.ABORT,
            reason=self.reason,
            safety_flags=safety_flags,
        )
        return setpoint, status

    def _build_setpoint(self, estimate: Estimate, metrics: dict[str, float]) -> GuidanceSetpoint:
        desired_position = np.array(
            [self.config.hold_point_x_m, 0.0, 0.0],
            dtype=float,
        )
        desired_velocity = np.zeros(3, dtype=float)
        far_approach_speed = 0.16
        final_approach_speed = min(self.config.final_approach_speed_mps, 0.03)

        if self.phase == self.ACQUIRE:
            self.final_target_x = None
            desired_position = estimate.relative_position.copy()
        elif self.phase == self.FAR_RANGE:
            self.final_target_x = None
            next_x = max(
                self.config.hold_point_x_m,
                metrics["axial_distance"] - far_approach_speed * self.config.control_period,
            )
            desired_position = np.array([next_x, 0.0, 0.0], dtype=float)
            desired_velocity = np.array([-far_approach_speed, 0.0, 0.0], dtype=float)
        elif self.phase == self.ALIGN:
            self.final_target_x = None
            desired_position = np.array([self.config.hold_point_x_m, 0.0, 0.0], dtype=float)
        elif self.phase == self.FINAL:
            if self.final_target_x is None:
                self.final_target_x = metrics["axial_distance"]
            if (
                metrics["lateral_error"] > 0.06
                or metrics["orientation_error_deg"] > 5.0
                or metrics["closing_speed"] > 0.08
            ):
                desired_position = np.array([self.final_target_x, 0.0, 0.0], dtype=float)
                desired_velocity = np.zeros(3, dtype=float)
            else:
                self.final_target_x = max(
                    self.config.capture_radius_m * 0.3,
                    self.final_target_x - final_approach_speed * self.config.control_period,
                )
                desired_position = np.array([self.final_target_x, 0.0, 0.0], dtype=float)
                desired_velocity = np.array([-final_approach_speed, 0.0, 0.0], dtype=float)
        elif self.phase == self.CAPTURE:
            self.final_target_x = None
            desired_position = np.zeros(3, dtype=float)
        elif self.phase == self.ABORT:
            self.final_target_x = None
            desired_position = np.array([self.config.retreat_point_x_m, 0.0, 0.0], dtype=float)

        return GuidanceSetpoint(
            desired_position=desired_position,
            desired_velocity=desired_velocity,
            desired_orientation=self.config.desired_docking_orientation.copy(),
        )

    def _compute_metrics(self, estimate: Estimate) -> dict[str, float]:
        position = estimate.relative_position
        velocity = estimate.relative_velocity
        axial_distance = max(position[0], 0.0)
        return {
            "distance": float(np.linalg.norm(position)),
            "axial_distance": float(axial_distance),
            "lateral_error": float(np.linalg.norm(position[1:])),
            "orientation_error_deg": math.degrees(
                angle_between_quaternions(
                    estimate.relative_orientation,
                    self.config.desired_docking_orientation,
                )
            ),
            "closing_speed": float(max(0.0, -velocity[0])),
            "angular_rate_deg_s": float(np.degrees(np.linalg.norm(estimate.relative_angular_velocity))),
            "speed_norm": float(np.linalg.norm(velocity)),
        }

    def _build_safety_flags(
        self,
        metrics: dict[str, float],
        estimate: Estimate,
        timestamp: float,
    ) -> dict[str, bool]:
        stale_time = (
            self.config.abort_sensor_timeout_s + 1.0
            if self.last_valid_time is None
            else timestamp - self.last_valid_time
        )
        return {
            "sensor_timeout": stale_time > self.config.abort_sensor_timeout_s or not estimate.valid,
            "lateral_limit": metrics["lateral_error"] > self.config.abort_lateral_limit_m,
            "angular_limit": metrics["orientation_error_deg"] > self.config.abort_angular_limit_deg,
            "closing_speed_limit": metrics["closing_speed"] > max(self.config.abort_closing_speed_mps, 0.18),
        }

    def _is_docked(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["distance"] <= self.config.capture_radius_m
            and metrics["lateral_error"] <= self.config.alignment_tolerance_m
            and metrics["orientation_error_deg"] <= self.config.angular_tolerance_deg
            and metrics["closing_speed"] <= self.config.closing_speed_limit_mps
            and metrics["angular_rate_deg_s"] <= self.config.angular_rate_limit_deg_s
        )

    def _close_range_violation(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["axial_distance"] <= 1.2
            and (
                metrics["lateral_error"] > max(self.config.abort_lateral_limit_m, 0.18)
                or metrics["orientation_error_deg"] > max(self.config.abort_angular_limit_deg, 18.0)
                or metrics["closing_speed"] > max(self.config.abort_closing_speed_mps, 0.16)
            )
        )
