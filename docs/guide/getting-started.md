<!-- Erwin Lejeune — 2026-02-23 -->
# Getting Started

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Installation

```bash
git clone https://github.com/guilyx/autonomous-quadrotor-guide.git
cd autonomous-quadrotor-guide
uv sync
```

## Running a Simulation

Every simulation lives under `src/uav_sim/simulations/<category>/<name>/` and can be run as a Python module:

```bash
python -m uav_sim.simulations.estimation.ekf
```

This produces a GIF animation and a JSON log file in the simulation directory.

## Building a Simulation Programmatically

For open-source users integrating their own algorithms, use the composition API:

```python
from pathlib import Path

from uav_sim.simulations import (
    StaticPathPlanner,
    create_environment,
    create_mission,
    create_sim,
    run_sim,
    spawn_quad_platform,
)
from uav_sim.simulations.common import figure_8_path
from uav_sim.simulations.standards import SimulationStandard

standard = SimulationStandard.flight_coupled()
planner = StaticPathPlanner(
    figure_8_path(duration=standard.duration, dt=0.15, alt=12.0, alt_amp=0.0, rx=8.0, ry=6.0)
)
env = create_environment(n_buildings=0, world_size=30.0)
platform = spawn_quad_platform()
mission = create_mission(
    path=planner.plan(world=env.world, standard=standard),
    standard=standard,
    fallback_policy="none",
)
sim = create_sim(name="my_first_sim", out_dir=Path("."), mission=mission)
result = run_sim(sim=sim, env=env, platform=platform)
print(result.mission.completion.timeout_reason)
```

## Running Tests

```bash
uv run pytest tests/
```

## Project Structure

```
src/uav_sim/
├── vehicles/          # 6DOF dynamics (Quadrotor, Fixed-Wing, VTOL)
├── control/           # Cascaded PID, State Machine
├── sensors/           # GPS, IMU, Lidar, Camera, Gimbal
├── estimation/        # EKF, UKF, Complementary, Particle Filter
├── perception/        # Occupancy mapping, SLAM, visual servoing
├── path_planning/     # A*, RRT*, PRM, Coverage
├── path_tracking/     # PID, LQR, Pure Pursuit
├── trajectory_planning/  # Min-Snap, Polynomial, Frenet
├── trajectory_tracking/  # Feedback Lin., MPPI, NMPC
├── swarm/             # Reynolds, Consensus, Virtual Structure
├── costmap/           # Occupancy grid, inflation layers
├── environment/       # World, obstacles, buildings
├── visualization/     # 3-panel viz, vehicle artists
└── simulations/       # 40+ runnable demos
```
