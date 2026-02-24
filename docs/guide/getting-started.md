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
