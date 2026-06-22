from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import quat_to_rotmat
from spacecraft_docking.types import SimulationResult


BackdropName = str


def save_summary_plots(
    result: SimulationResult,
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / f"{result.scenario_name}_metrics.png"
    trajectory_file = output_path / f"{result.scenario_name}_trajectory.png"

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.patch.set_facecolor("#060911")
    for axis in axes:
        axis.set_facecolor("#0a1020")
        axis.tick_params(colors="#f2f4ff")
        for spine in axis.spines.values():
            spine.set_color("#8ea1c9")
        axis.grid(True, alpha=0.25, color="#8ea1c9")

    axes[0].plot(result.times, result.position_error_norm, color="#6bd0ff", linewidth=2.2)
    axes[0].set_ylabel("Rel Dist (m)", color="#f2f4ff")
    axes[0].set_title("Moving-Target Docking Telemetry", color="#f2f4ff")

    axes[1].plot(result.times, result.closing_speed, color="#85f2b1", linewidth=2.2, label="Closing speed")
    axes[1].plot(result.times, np.linalg.norm(result.relative_velocities[:, 1:], axis=1), color="#ffd166", linewidth=1.8, label="Lateral speed")
    axes[1].set_ylabel("Rel Vel (m/s)", color="#f2f4ff")
    axes[1].legend(facecolor="#0c1324", edgecolor="#8ea1c9", labelcolor="#f2f4ff")

    axes[2].plot(result.times, result.orientation_error_deg, color="#ff9861", linewidth=2.2, label="Orientation error")
    axes[2].plot(result.times, np.degrees(np.linalg.norm(result.relative_angular_velocities, axis=1)), color="#c3a6ff", linewidth=1.8, label="Relative ang rate")
    axes[2].set_ylabel("Angle / Rate", color="#f2f4ff")
    axes[2].legend(facecolor="#0c1324", edgecolor="#8ea1c9", labelcolor="#f2f4ff")

    axes[3].plot(result.times, np.linalg.norm(result.command_forces_body, axis=1), color="#f472b6", linewidth=2.0, label="Thruster force")
    axes[3].plot(result.times, result.chaser_angular_momentum_norm, color="#7dd3fc", linewidth=1.8, label="Chaser ang momentum")
    axes[3].set_ylabel("Control / H", color="#f2f4ff")
    axes[3].set_xlabel("Time (s)", color="#f2f4ff")
    axes[3].legend(facecolor="#0c1324", edgecolor="#8ea1c9", labelcolor="#f2f4ff")
    fig.tight_layout()
    fig.savefig(metrics_file, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 7), facecolor="#060911")
    _add_background_layer(fig, backdrop="earth")
    axis = fig.add_subplot(111, projection="3d", facecolor=(0.0, 0.0, 0.0, 0.0))
    _apply_scene_style(axis, result)
    _draw_station(axis, result.target_positions[-1], result.target_orientations[-1])
    _draw_chaser(axis, result.positions[-1], result.orientations[-1])
    axis.plot(
        result.positions[:, 0],
        result.positions[:, 1],
        result.positions[:, 2],
        color="#5ad5ff",
        linewidth=2.4,
        alpha=0.95,
        label="Chaser path",
    )
    axis.plot(
        result.target_positions[:, 0],
        result.target_positions[:, 1],
        result.target_positions[:, 2],
        color="#ffb454",
        linewidth=2.0,
        alpha=0.9,
        linestyle="--",
        label="Station path",
    )
    axis.scatter(
        [result.positions[0, 0]],
        [result.positions[0, 1]],
        [result.positions[0, 2]],
        color="#ffd166",
        s=70,
        label="Chaser start",
    )
    axis.scatter(
        [result.target_positions[0, 0]],
        [result.target_positions[0, 1]],
        [result.target_positions[0, 2]],
        color="#ff7b54",
        s=70,
        label="Station start",
    )
    _draw_planet_label(axis, "EARTH")
    axis.set_title(f"Relative Docking in Moving Frame: {result.scenario_name}", color="#f2f4ff", pad=18)
    axis.legend(loc="upper right", facecolor="#0c1324", edgecolor="#8ea1c9", labelcolor="#f2f4ff")
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.93)
    fig.savefig(trajectory_file, dpi=180)
    plt.close(fig)

    return [metrics_file, trajectory_file]


def save_animation(
    result: SimulationResult,
    config: SimulationConfig,
    output_path: str | Path,
    frame_stride: int = 8,
    backdrop: BackdropName = "earth",
) -> Path:
    animation_path = Path(output_path)
    animation_path.parent.mkdir(parents=True, exist_ok=True)
    fig, animation = _build_animation(result, config, frame_stride=frame_stride, backdrop=backdrop)
    writer = "pillow" if animation_path.suffix.lower() == ".gif" else None
    animation.save(animation_path, writer=writer, dpi=120)
    plt.close(fig)
    return animation_path


def show_live_animation(
    result: SimulationResult,
    config: SimulationConfig,
    frame_stride: int = 4,
    backdrop: BackdropName = "earth",
) -> None:
    fig, _ = _build_animation(result, config, frame_stride=frame_stride, backdrop=backdrop)
    plt.show()
    plt.close(fig)


def _build_animation(
    result: SimulationResult,
    config: SimulationConfig,
    frame_stride: int,
    backdrop: BackdropName,
) -> tuple[plt.Figure, FuncAnimation]:
    fig = plt.figure(figsize=(13, 7), facecolor="#050812")
    _add_background_layer(fig, backdrop=backdrop)

    axis = fig.add_axes([0.04, 0.08, 0.64, 0.84], projection="3d", facecolor=(0.0, 0.0, 0.0, 0.0))
    _apply_scene_style(axis, result)

    telemetry_axis = fig.add_axes([0.71, 0.10, 0.26, 0.80])
    telemetry_axis.axis("off")
    telemetry_axis.set_facecolor((0.02, 0.03, 0.08, 0.65))
    telemetry_text = telemetry_axis.text(
        0.03,
        0.97,
        "",
        va="top",
        ha="left",
        color="#eef4ff",
        fontsize=10.5,
        family="monospace",
        linespacing=1.4,
    )

    chaser_path_line, = axis.plot([], [], [], color="#5ad5ff", linewidth=2.4, alpha=0.95)
    target_path_line, = axis.plot([], [], [], color="#ffb454", linewidth=1.8, alpha=0.85, linestyle="--")
    _draw_planet_label(axis, backdrop.upper() if backdrop != "none" else "SPACE")

    frame_indices = list(range(0, len(result.times), max(1, frame_stride)))
    if frame_indices[-1] != len(result.times) - 1:
        frame_indices.append(len(result.times) - 1)

    dynamic_artists: list = []

    def update(frame_number: int):
        nonlocal dynamic_artists
        for artist in dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        dynamic_artists = []

        index = frame_indices[frame_number]
        chaser_path_line.set_data(result.positions[: index + 1, 0], result.positions[: index + 1, 1])
        chaser_path_line.set_3d_properties(result.positions[: index + 1, 2])
        target_path_line.set_data(result.target_positions[: index + 1, 0], result.target_positions[: index + 1, 1])
        target_path_line.set_3d_properties(result.target_positions[: index + 1, 2])

        dynamic_artists.extend(_draw_station(axis, result.target_positions[index], result.target_orientations[index]))
        dynamic_artists.extend(_draw_docking_corridor(axis, result.target_positions[index], result.target_orientations[index]))
        dynamic_artists.extend(_draw_chaser(axis, result.positions[index], result.orientations[index]))

        axis.set_title(
            f"Live Relative Docking Simulation: {result.scenario_name} | "
            f"t={result.times[index]:.1f}s | phase={result.phases[index]}",
            color="#f2f4ff",
            pad=16,
        )
        telemetry_text.set_text(_format_telemetry(result, index))
        return [chaser_path_line, target_path_line, telemetry_text, *dynamic_artists]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=max(25, int(config.dt_physics * frame_stride * 1000)),
        blit=False,
        repeat=False,
    )
    return fig, animation


def _format_telemetry(result: SimulationResult, index: int) -> str:
    rel_vel_norm = np.linalg.norm(result.relative_velocities[index])
    rel_rate_deg_s = np.degrees(np.linalg.norm(result.relative_angular_velocities[index]))
    thrust_norm = np.linalg.norm(result.command_forces_body[index])
    torque_norm = np.linalg.norm(result.command_torques_body[index])
    return "\n".join(
        [
            "Telemetry",
            "------------------------------",
            f"Distance        : {result.position_error_norm[index]:6.3f} m",
            f"Lateral error   : {result.lateral_error_norm[index]:6.3f} m",
            f"Closing speed   : {result.closing_speed[index]:6.3f} m/s",
            f"Rel speed       : {rel_vel_norm:6.3f} m/s",
            f"Orientation err : {result.orientation_error_deg[index]:6.3f} deg",
            f"Rel ang rate    : {rel_rate_deg_s:6.3f} deg/s",
            "",
            f"Chaser speed    : {result.chaser_speed[index]:6.3f} m/s",
            f"Station speed   : {result.target_speed[index]:6.3f} m/s",
            f"Chaser |H|      : {result.chaser_angular_momentum_norm[index]:6.3f}",
            f"Station |H|     : {result.target_angular_momentum_norm[index]:6.3f}",
            "",
            f"Thruster force  : {thrust_norm:6.3f} N",
            f"Thruster torque : {torque_norm:6.3f} Nm",
            "",
            f"Rel pos X/Y/Z   : {result.relative_positions[index][0]:6.3f},"
            f" {result.relative_positions[index][1]:6.3f},"
            f" {result.relative_positions[index][2]:6.3f}",
            f"Phase           : {result.phases[index]}",
        ]
    )


def _apply_scene_style(axis, result: SimulationResult) -> None:
    combined = np.vstack((result.positions, result.target_positions))
    min_xyz = np.min(combined, axis=0)
    max_xyz = np.max(combined, axis=0)

    padding = np.array([1.4, 2.2, 2.2], dtype=float)
    axis.set_xlim(min_xyz[0] - 0.8, max_xyz[0] + padding[0])
    axis.set_ylim(min_xyz[1] - padding[1], max_xyz[1] + padding[1])
    axis.set_zlim(min_xyz[2] - padding[2], max_xyz[2] + padding[2])
    axis.view_init(elev=24, azim=-58)
    axis.set_box_aspect(
        (
            max(4.5, (max_xyz[0] - min_xyz[0]) + 2.0),
            max(4.5, (max_xyz[1] - min_xyz[1]) + 4.0),
            max(3.5, (max_xyz[2] - min_xyz[2]) + 4.0),
        )
    )

    axis.xaxis.pane.set_facecolor((0.02, 0.03, 0.08, 0.05))
    axis.yaxis.pane.set_facecolor((0.02, 0.03, 0.08, 0.02))
    axis.zaxis.pane.set_facecolor((0.02, 0.03, 0.08, 0.02))
    axis.xaxis.pane.set_edgecolor((0.7, 0.78, 1.0, 0.10))
    axis.yaxis.pane.set_edgecolor((0.7, 0.78, 1.0, 0.08))
    axis.zaxis.pane.set_edgecolor((0.7, 0.78, 1.0, 0.08))
    axis.set_facecolor((0.0, 0.0, 0.0, 0.0))
    axis.grid(True, color="#8ea1c9", alpha=0.16)

    axis.tick_params(colors="#d7def5")
    axis.xaxis.label.set_color("#eaf0ff")
    axis.yaxis.label.set_color("#eaf0ff")
    axis.zaxis.label.set_color("#eaf0ff")
    axis.set_xlabel("World X (m)", labelpad=10)
    axis.set_ylabel("World Y (m)", labelpad=10)
    axis.set_zlabel("World Z (m)", labelpad=10)


def _add_background_layer(fig: plt.Figure, backdrop: BackdropName) -> None:
    background_axis = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
    background_axis.imshow(_generate_background_image(backdrop), aspect="auto", interpolation="bilinear")
    background_axis.axis("off")


def _generate_background_image(backdrop: BackdropName, height: int = 900, width: int = 1600) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height)[:, None]
    x = np.linspace(0.0, 1.0, width)[None, :]
    xx = np.broadcast_to(x, (height, width))
    yy = np.broadcast_to(y, (height, width))
    image = np.zeros((height, width, 3), dtype=float)

    top = np.array([4, 7, 18], dtype=float) / 255.0
    bottom = np.array([1, 2, 7], dtype=float) / 255.0
    image[:] = bottom + (top - bottom) * (1.0 - y[..., None])

    rng = np.random.default_rng(12)
    stars = rng.random((height, width))
    bright = stars > 0.9984
    medium = (stars > 0.9965) & ~bright
    image[medium] = np.maximum(image[medium], np.array([0.60, 0.68, 0.95]))
    image[bright] = np.array([0.95, 0.97, 1.0])

    if backdrop != "none":
        palette = _planet_palette(backdrop)
        cx = 0.84
        cy = 0.18
        radius = 0.26
        dx = xx - cx
        dy = yy - cy
        distance = np.sqrt(dx * dx + dy * dy)
        mask = distance <= radius
        if np.any(mask):
            radial = np.clip(distance[mask] / radius, 0.0, 1.0)
            light = np.clip(1.1 - 1.8 * radial + 0.6 * (-dx[mask] / radius), 0.0, 1.0)
            atmosphere = np.clip(1.0 - radial, 0.0, 1.0)
            planet = (
                palette["dark"][None, :] * (1.0 - light[:, None])
                + palette["light"][None, :] * light[:, None]
            )
            planet = np.clip(planet + palette["accent"][None, :] * atmosphere[:, None] * 0.18, 0.0, 1.0)
            image[mask] = planet

            ring_mask = (distance > radius) & (distance <= radius * 1.08)
            image[ring_mask] = np.clip(image[ring_mask] + palette["accent"][None, :] * 0.32, 0.0, 1.0)
    return image


def _planet_palette(backdrop: BackdropName) -> dict[str, np.ndarray]:
    palettes: dict[BackdropName, dict[str, tuple[int, int, int]]] = {
        "earth": {"dark": (25, 53, 112), "light": (133, 208, 255), "accent": (92, 255, 192)},
        "mars": {"dark": (96, 41, 23), "light": (230, 151, 103), "accent": (255, 189, 135)},
        "neptune": {"dark": (29, 51, 133), "light": (116, 173, 255), "accent": (144, 233, 255)},
    }
    raw = palettes.get(backdrop, palettes["earth"])
    return {key: np.array(value, dtype=float) / 255.0 for key, value in raw.items()}


def _draw_station(axis, position: np.ndarray, orientation: np.ndarray) -> list:
    rot = quat_to_rotmat(orientation)
    artists = [
        _add_box(axis, center=position, size=np.array([0.65, 0.55, 0.55]), color="#bcc8e4", alpha=0.96, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([-0.62, 0.0, 0.0]), size=np.array([0.55, 0.34, 0.34]), color="#90a4c6", alpha=0.92, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.62, 0.0, 0.0]), size=np.array([0.38, 0.30, 0.30]), color="#d5dff0", alpha=0.98, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.0, 0.68, 0.0]), size=np.array([0.24, 0.62, 0.24]), color="#8798b8", alpha=0.90, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.0, -0.68, 0.0]), size=np.array([0.24, 0.62, 0.24]), color="#8798b8", alpha=0.90, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([-0.40, 1.30, 0.0]), size=np.array([0.10, 1.16, 0.03]), color="#2f5fcb", alpha=0.84, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.40, 1.30, 0.0]), size=np.array([0.10, 1.16, 0.03]), color="#2f5fcb", alpha=0.84, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([-0.40, -1.30, 0.0]), size=np.array([0.10, 1.16, 0.03]), color="#2f5fcb", alpha=0.84, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.40, -1.30, 0.0]), size=np.array([0.10, 1.16, 0.03]), color="#2f5fcb", alpha=0.84, rotation=rot),
    ]

    port_center = position + rot @ np.array([0.72, 0.0, 0.0])
    artists.extend(_draw_ring(axis, center=port_center, rotation=rot, radius=0.16, color="#ffb454", linewidth=2.2))
    return artists


def _draw_docking_corridor(axis, target_position: np.ndarray, target_orientation: np.ndarray) -> list:
    rot = quat_to_rotmat(target_orientation)
    port_center = target_position + rot @ np.array([0.72, 0.0, 0.0])
    y_extent = 0.28
    z_extent = 0.28
    x_values = np.linspace(0.0, 6.0, 16)
    artists: list = []

    for y_sign, z_sign in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        points = np.array([[x, y_sign * y_extent, z_sign * z_extent] for x in x_values], dtype=float)
        points_world = port_center + points @ rot.T
        line, = axis.plot(points_world[:, 0], points_world[:, 1], points_world[:, 2], color="#50d6ff", linewidth=1.25, alpha=0.16)
        artists.append(line)

    for x_value in x_values[::2]:
        loop = np.array(
            [
                [x_value, y_extent, z_extent],
                [x_value, -y_extent, z_extent],
                [x_value, -y_extent, -z_extent],
                [x_value, y_extent, -z_extent],
                [x_value, y_extent, z_extent],
            ],
            dtype=float,
        )
        loop_world = port_center + loop @ rot.T
        line, = axis.plot(loop_world[:, 0], loop_world[:, 1], loop_world[:, 2], color="#50d6ff", linewidth=0.9, alpha=0.11)
        artists.append(line)
    return artists


def _draw_chaser(axis, position: np.ndarray, orientation: np.ndarray) -> list:
    rot = quat_to_rotmat(orientation)
    artists = [
        _add_box(axis, center=position, size=np.array([0.62, 0.30, 0.24]), color="#d9e2f2", alpha=0.97, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([-0.12, 0.0, 0.0]), size=np.array([0.28, 0.20, 0.18]), color="#9faecc", alpha=0.94, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.34, 0.0, 0.0]), size=np.array([0.14, 0.12, 0.12]), color="#f4a259", alpha=0.95, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.0, 0.42, 0.0]), size=np.array([0.18, 0.62, 0.02]), color="#4b7cff", alpha=0.88, rotation=rot),
        _add_box(axis, center=position + rot @ np.array([0.0, -0.42, 0.0]), size=np.array([0.18, 0.62, 0.02]), color="#4b7cff", alpha=0.88, rotation=rot),
    ]
    nose_ring = position + rot @ np.array([0.42, 0.0, 0.0])
    scatter = axis.scatter([nose_ring[0]], [nose_ring[1]], [nose_ring[2]], color="#ffd166", s=42, depthshade=False)
    artists.append(scatter)
    thruster_tail = np.array([position, position - rot @ np.array([0.55, 0.0, 0.0])])
    line, = axis.plot(thruster_tail[:, 0], thruster_tail[:, 1], thruster_tail[:, 2], color="#ffd166", linewidth=1.4, alpha=0.88)
    artists.append(line)
    return artists


def _add_box(
    axis,
    center: np.ndarray,
    size: np.ndarray,
    color: str,
    alpha: float,
    rotation: np.ndarray | None = None,
) -> Poly3DCollection:
    hx, hy, hz = size / 2.0
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    if rotation is not None:
        vertices = vertices @ rotation.T
    vertices = vertices + center

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
    ]
    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors=(1.0, 1.0, 1.0, min(0.18, alpha)),
        linewidths=0.7,
        alpha=alpha,
    )
    axis.add_collection3d(poly)
    return poly


def _draw_ring(axis, center: np.ndarray, rotation: np.ndarray, radius: float, color: str, linewidth: float) -> list:
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    local_points = np.stack(
        [
            np.zeros_like(theta),
            radius * np.cos(theta),
            radius * np.sin(theta),
        ],
        axis=1,
    )
    world_points = center + local_points @ rotation.T
    line, = axis.plot(world_points[:, 0], world_points[:, 1], world_points[:, 2], color=color, linewidth=linewidth, alpha=0.95)
    return [line]


def _draw_planet_label(axis, label: str):
    return axis.text2D(
        0.86,
        0.91,
        label,
        transform=axis.transAxes,
        color="#d7e7ff",
        fontsize=11,
        ha="center",
        va="center",
        alpha=0.85,
    )
