from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path

import numpy as np

from scenarios import get_scenario, list_scenarios
from simulation import DockingSimulation
from spacecraft_docking.config import SimulationConfig
from spacecraft_docking.math_utils import quat_from_euler, quat_to_euler
from spacecraft_docking.types import RigidBodyState, Scenario
from visualization import save_animation, save_summary_plots, show_live_animation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spacecraft autodocking simulator")
    parser.add_argument(
        "--scenario",
        default="nominal",
        choices=list_scenarios(),
        help="Scenario to simulate",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for noise generation")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for plots and animation files",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Only save telemetry plots",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open a live viewer window after the simulation completes",
    )
    parser.add_argument(
        "--backdrop",
        default="earth",
        choices=["earth", "mars", "neptune", "none"],
        help="Backdrop planet for the cinematic scene",
    )
    parser.add_argument("--duration", type=float, help="Override scenario duration in seconds")
    parser.add_argument("--start-x", type=float, help="Override chaser start X position (m)")
    parser.add_argument("--start-y", type=float, help="Override chaser start Y position (m)")
    parser.add_argument("--start-z", type=float, help="Override chaser start Z position (m)")
    parser.add_argument("--start-vx", type=float, help="Override chaser start X velocity (m/s)")
    parser.add_argument("--start-vy", type=float, help="Override chaser start Y velocity (m/s)")
    parser.add_argument("--start-vz", type=float, help="Override chaser start Z velocity (m/s)")
    parser.add_argument("--start-roll", type=float, help="Override chaser roll (deg)")
    parser.add_argument("--start-pitch", type=float, help="Override chaser pitch (deg)")
    parser.add_argument("--start-yaw", type=float, help="Override chaser yaw (deg)")
    parser.add_argument("--target-x", type=float, help="Override station start X position (m)")
    parser.add_argument("--target-y", type=float, help="Override station start Y position (m)")
    parser.add_argument("--target-z", type=float, help="Override station start Z position (m)")
    parser.add_argument("--target-vx", type=float, help="Override station X velocity (m/s)")
    parser.add_argument("--target-vy", type=float, help="Override station Y velocity (m/s)")
    parser.add_argument("--target-vz", type=float, help="Override station Z velocity (m/s)")
    parser.add_argument("--target-roll", type=float, help="Override station roll (deg)")
    parser.add_argument("--target-pitch", type=float, help="Override station pitch (deg)")
    parser.add_argument("--target-yaw", type=float, help="Override station yaw (deg)")
    parser.add_argument("--target-wx", type=float, help="Override station roll rate (deg/s)")
    parser.add_argument("--target-wy", type=float, help="Override station pitch rate (deg/s)")
    parser.add_argument("--target-wz", type=float, help="Override station yaw rate (deg/s)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimulationConfig()
    simulation = DockingSimulation(config)
    scenario = _scenario_with_overrides(get_scenario(args.scenario), args)

    result = simulation.run(scenario, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = save_summary_plots(result, output_dir)

    if args.live:
        print("Opening live viewer window. Close it to continue.")
        show_live_animation(result, config, backdrop=args.backdrop)

    if args.live and not args.skip_animation:
        print("Live mode finished. GIF export skipped; rerun without --live if you want to save the animation.")
    elif not args.skip_animation:
        try:
            animation_file = save_animation(
                result,
                config,
                output_dir / f"{scenario.name}_animation.gif",
                backdrop=args.backdrop,
            )
            saved_files.append(animation_file)
        except Exception as exc:  # pragma: no cover - convenience for local runs
            print(f"Animation export skipped: {exc}")

    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Final phase: {result.final_status.phase}")
    print(f"Docked: {result.final_status.docked}")
    print(f"Aborted: {result.final_status.aborted}")
    print(f"Reason: {result.final_status.reason}")
    for metric_name, metric_value in result.metrics.items():
        print(f"{metric_name}: {metric_value:.3f}")
    print("Saved artifacts:")
    for file_path in saved_files:
        print(f"  - {file_path}")


def _scenario_with_overrides(base_scenario: Scenario, args: argparse.Namespace) -> Scenario:
    scenario = replace(
        base_scenario,
        initial_chaser_state=_clone_state(base_scenario.initial_chaser_state),
        initial_target_state=_clone_state(base_scenario.initial_target_state),
    )

    if args.duration is not None:
        scenario.duration = args.duration

    scenario.initial_chaser_state = _override_state(
        scenario.initial_chaser_state,
        position=(args.start_x, args.start_y, args.start_z),
        velocity=(args.start_vx, args.start_vy, args.start_vz),
        euler_deg=(args.start_roll, args.start_pitch, args.start_yaw),
        angular_rate_deg_s=(None, None, None),
    )
    scenario.initial_target_state = _override_state(
        scenario.initial_target_state,
        position=(args.target_x, args.target_y, args.target_z),
        velocity=(args.target_vx, args.target_vy, args.target_vz),
        euler_deg=(args.target_roll, args.target_pitch, args.target_yaw),
        angular_rate_deg_s=(args.target_wx, args.target_wy, args.target_wz),
    )
    return scenario


def _clone_state(state: RigidBodyState) -> RigidBodyState:
    return replace(
        state,
        position=state.position.copy(),
        velocity=state.velocity.copy(),
        orientation=state.orientation.copy(),
        angular_velocity=state.angular_velocity.copy(),
    )


def _override_state(
    state: RigidBodyState,
    position: tuple[float | None, float | None, float | None],
    velocity: tuple[float | None, float | None, float | None],
    euler_deg: tuple[float | None, float | None, float | None],
    angular_rate_deg_s: tuple[float | None, float | None, float | None],
) -> RigidBodyState:
    next_state = _clone_state(state)

    for index, value in enumerate(position):
        if value is not None:
            next_state.position[index] = value

    for index, value in enumerate(velocity):
        if value is not None:
            next_state.velocity[index] = value

    if any(value is not None for value in euler_deg):
        current_euler = tuple(math.degrees(value) for value in quat_to_euler(next_state.orientation))
        roll, pitch, yaw = (
            euler_deg[index] if euler_deg[index] is not None else current_euler[index]
            for index in range(3)
        )
        next_state.orientation = quat_from_euler(
            math.radians(roll),
            math.radians(pitch),
            math.radians(yaw),
        )

    for index, value in enumerate(angular_rate_deg_s):
        if value is not None:
            next_state.angular_velocity[index] = math.radians(value)

    return next_state


if __name__ == "__main__":
    main()
