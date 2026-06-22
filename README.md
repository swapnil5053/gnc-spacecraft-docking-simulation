# Spacecraft Docking Simulation

A GNC-based spacecraft docking simulator with a web-based autopilot interface developed as part of a Mobile Autonomous Robotics (MAR) course project.

## Overview

This project explores autonomous spacecraft docking in a simulated microgravity environment.

The simulation models how a spacecraft can detect a target vehicle, estimate its relative position and orientation, plan a safe approach, and complete a docking maneuver without manual intervention. To complement the simulation, the project also includes a web-based autopilot dashboard for visualizing spacecraft behavior and mission data.

The repository combines simulation, control systems, and interactive visualization into a single project focused on autonomous docking operations.

## Features

* Autonomous spacecraft docking simulation
* Guidance, Navigation, and Control (GNC) pipeline
* Relative state estimation and tracking
* Sensor noise and fault testing
* Multiple mission scenarios
* Real-time telemetry visualization
* Interactive autopilot dashboard
* Simulation outputs including plots and animations

## Repository Structure

```text
gnc-spacecraft-docking-simulation
│
├── run_simulation.py
├── simulation.py
├── physics.py
├── sensors.py
├── estimation.py
├── guidance.py
├── control.py
├── scenarios.py
│
├── outputs/
│
├── REPORT.md
│
└── spacecraft-autopilot/
```

## Running the Simulation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the nominal simulation:

```bash
python run_simulation.py --scenario nominal
```

Run other scenarios:

```bash
python run_simulation.py --scenario offset_start --seed 11

python run_simulation.py --scenario nominal --live --skip-animation --backdrop earth
```

Run the test suite:

```bash
python -m unittest test_simulation.py
```

## Running the Dashboard

Navigate to the frontend application:

```bash
cd spacecraft-autopilot
```

Install dependencies:

```bash
pnpm install
```

Start the development server:

```bash
pnpm run dev
```

Additional tuning utilities:

```bash
pnpm run tune

pnpm run tune:optimize
```

## Technologies Used

### Simulation

* Python
* NumPy
* Matplotlib

### Dashboard

* React
* TypeScript
* Vite
* Three.js
* Rapier Physics

## Documentation

Detailed information about the system architecture, docking workflow, guidance logic, controller design, evaluation scenarios, assumptions, limitations, and future improvements can be found in `REPORT.md`.
## Contributors

Developed by:

- **[Rishi Kumar](https://github.com/Rishi94523)**
- **[Swapnil Kumar](https://github.com/swapnil5053)**
