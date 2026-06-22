from __future__ import annotations

import math

import numpy as np


EPSILON = 1e-9


def vec3(values: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < EPSILON:
        return np.zeros_like(vec, dtype=float)
    return vec / norm


def clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm < EPSILON:
        return vec
    return vec * (max_norm / norm)


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < EPSILON:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([w, -x, -y, -z], dtype=float)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    if np.linalg.norm(axis) < EPSILON:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    half = angle_rad / 2.0
    sin_half = math.sin(half)
    return quat_normalize(
        np.array([math.cos(half), *(axis * sin_half)], dtype=float)
    )


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    return quat_normalize(quat)


def quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat_normalize(quat)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(quat) @ vec


def nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    blend = (1.0 - alpha) * q0 + alpha * q1
    return quat_normalize(blend)


def integrate_quaternion(quat: np.ndarray, angular_velocity_body: np.ndarray, dt: float) -> np.ndarray:
    omega_quat = np.array([0.0, *angular_velocity_body], dtype=float)
    q_dot = 0.5 * quat_multiply(quat, omega_quat)
    return quat_normalize(quat + q_dot * dt)


def quat_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return quat_normalize(quat_multiply(quat_conjugate(current), desired))


def quat_to_axis_angle(quat: np.ndarray) -> tuple[np.ndarray, float]:
    quat = quat_normalize(quat)
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
    axis = quat[1:] / s
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return normalize(axis), angle


def angle_between_quaternions(current: np.ndarray, desired: np.ndarray) -> float:
    _, angle = quat_to_axis_angle(quat_error(current, desired))
    return abs(angle)
