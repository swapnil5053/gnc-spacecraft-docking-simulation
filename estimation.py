from __future__ import annotations

from collections import deque

import numpy as np

from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import nlerp
from spacecraft_docking.types import Estimate, SensorPacket


class RelativeStateEstimator:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.estimate = Estimate(
            relative_position=np.zeros(3, dtype=float),
            relative_velocity=np.zeros(3, dtype=float),
            relative_orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            relative_angular_velocity=np.zeros(3, dtype=float),
            confidence=0.0,
            valid=False,
        )
        self.last_measurement_position: np.ndarray | None = None
        self.last_measurement_time: float | None = None
        self.position_history: deque[tuple[float, np.ndarray]] = deque(maxlen=8)

    def step(self, packet: SensorPacket | None, timestamp: float) -> Estimate:
        if packet is not None and packet.valid:
            self.position_history.append((packet.timestamp, packet.relative_position.copy()))
            position = (
                packet.relative_position
                if self.last_measurement_position is None
                else self.config.estimator_position_alpha * self.estimate.relative_position
                + (1.0 - self.config.estimator_position_alpha) * packet.relative_position
            )

            raw_velocity = np.zeros(3, dtype=float)
            if len(self.position_history) >= 2:
                oldest_time, oldest_position = self.position_history[0]
                newest_time, newest_position = self.position_history[-1]
                dt = max(newest_time - oldest_time, 1e-6)
                window_velocity = (newest_position - oldest_position) / dt
                raw_velocity = 0.35 * window_velocity + 0.65 * packet.relative_velocity
            else:
                raw_velocity = packet.relative_velocity

            if self.last_measurement_position is None:
                velocity = packet.relative_velocity
            else:
                velocity = (
                    self.config.estimator_velocity_alpha * self.estimate.relative_velocity
                    + (1.0 - self.config.estimator_velocity_alpha) * raw_velocity
                )

            orientation = nlerp(
                self.estimate.relative_orientation,
                packet.relative_orientation,
                1.0 - self.config.estimator_orientation_alpha,
            )

            self.estimate = Estimate(
                relative_position=position,
                relative_velocity=velocity,
                relative_orientation=orientation,
                relative_angular_velocity=packet.imu_rates,
                confidence=1.0,
                valid=True,
            )
            self.last_measurement_position = packet.relative_position.copy()
            self.last_measurement_time = packet.timestamp
            return self.estimate

        if self.last_measurement_time is None:
            self.estimate.valid = False
            self.estimate.confidence = 0.0
            return self.estimate

        stale_for = timestamp - self.last_measurement_time
        confidence = max(0.0, 1.0 - stale_for / self.config.abort_sensor_timeout_s)
        self.estimate = Estimate(
            relative_position=self.estimate.relative_position,
            relative_velocity=self.estimate.relative_velocity,
            relative_orientation=self.estimate.relative_orientation,
            relative_angular_velocity=self.estimate.relative_angular_velocity,
            confidence=confidence,
            valid=stale_for <= self.config.estimator_timeout_s,
        )
        return self.estimate
