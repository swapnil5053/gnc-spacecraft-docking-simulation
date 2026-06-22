# Spacecraft Docking Simulation

A Python microgravity simulator for a robotics subject project. The project models a chaser spacecraft autonomously docking with a passive target using a sensing -> estimation -> guidance -> control pipeline.

## What It Includes

- 3D rigid-body motion in a microgravity docking zone
- Moving and rotating target station with relative-motion docking
- Noisy relative-position/orientation sensing and IMU ratess
- Lightweight filtered state estimation
- Finite-state-machine autodocking logic
- PD translational and attitude control
- Scenario-based testing for nominal docking and safety aborts
- Saved trajectory, convergence plots, and optional animation export

## Run It

```bash
python run_simulation.py --scenario nominal
```

Useful options:

```bash
python run_simulation.py --scenario offset_start --seed 11
python run_simulation.py --scenario actuator_saturation --skip-animation
python run_simulation.py --scenario nominal --live --skip-animation --backdrop earth
python run_simulation.py --scenario nominal --backdrop mars
python run_simulation.py --scenario nominal --start-y 0.6 --start-z -0.25 --target-vy 0.01 --target-wz 0.05
```

Live mode notes:

- `--live` opens a cinematic viewer window and stops on the last frame
- `--skip-animation` is recommended with `--live` for a fast demo
- `--backdrop` accepts `earth`, `mars`, `neptune`, or `none`
- when `--live` is used, GIF export is skipped to avoid a second long wait
- start state and station motion can be overridden from the CLI with flags like `--start-x`, `--start-y`, `--start-z`, `--target-vx`, `--target-vy`, and `--target-wz`

Available scenarios:

- `nominal`
- `offset_start`
- `sensor_noise`
- `actuator_saturation`
- `unsafe_closing_speed`
- `large_misalignment`
- `no_sensor_lock`

## Test It

```bash
python -m unittest test_simulation.py
```

## Project Structure

- `simulation.py`: main loop and telemetry capture
- `physics.py`: microgravity rigid-body propagation
- `sensors.py`: noisy relative-navigation measurements
- `estimation.py`: filtered state estimate
- `guidance.py`: docking finite-state machine and safety logic
- `control.py`: PD position and attitude controllers
- `scenarios.py`: subject-project evaluation scenarios
- `visualization.py`: plots and GIF animation export
- `REPORT.md`: subject-ready explanation and diagrams
