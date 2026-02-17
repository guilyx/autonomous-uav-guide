# Autonomous Quadrotor Guide

[![CI](https://github.com/guilyx/autonomous-quadrotor-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/guilyx/autonomous-quadrotor-guide/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

A from-scratch Python library for learning autonomous quadrotor algorithms — from 6DOF dynamics and PID control to state estimation, path planning, trajectory optimisation, and multi-agent swarms. Every algorithm has a runnable simulation that produces an animated GIF.

## Quick Start

```bash
# clone & install (requires uv)
git clone https://github.com/guilyx/autonomous-quadrotor-guide.git
cd autonomous-quadrotor-guide
uv sync

# run any simulation
uv run python simulations/pid_hover/run.py

# run tests
uv run pytest
```

## Algorithms

### Control

| Algorithm | Simulation | Docs |
|-----------|-----------|------|
| **Cascaded PID** | ![PID Hover](simulations/pid_hover/pid_hover.gif) | [docs/control/pid.md](docs/control/pid.md) |
| **LQR** | ![LQR Hover](simulations/lqr_hover/lqr_hover.gif) | [docs/control/lqr.md](docs/control/lqr.md) |
| **Geometric SO(3)** | ![Geometric](simulations/geometric_control/geometric_control.gif) | [docs/control/geometric.md](docs/control/geometric.md) |

### State Estimation

| Algorithm | Simulation | Docs |
|-----------|-----------|------|
| **Complementary Filter** | ![CF](simulations/complementary_filter/complementary_filter.gif) | [docs/estimation/complementary_filter.md](docs/estimation/complementary_filter.md) |
| **Extended Kalman Filter** | ![EKF](simulations/ekf/ekf.gif) | [docs/estimation/ekf.md](docs/estimation/ekf.md) |
| **Unscented Kalman Filter** | ![UKF](simulations/ukf/ukf.gif) | [docs/estimation/ukf.md](docs/estimation/ukf.md) |
| **Particle Filter** | ![PF](simulations/particle_filter/particle_filter.gif) | [docs/estimation/particle_filter.md](docs/estimation/particle_filter.md) |

### Path Planning

| Algorithm | Simulation | Docs |
|-----------|-----------|------|
| **A\* 3D** | ![A*](simulations/astar_3d/astar_3d.gif) | [docs/planning/astar_3d.md](docs/planning/astar_3d.md) |
| **RRT\*** | ![RRT*](simulations/rrt_star_3d/rrt_star_3d.gif) | [docs/planning/rrt_star.md](docs/planning/rrt_star.md) |
| **Potential Field** | ![PF3D](simulations/potential_field_3d/potential_field_3d.gif) | [docs/planning/potential_field.md](docs/planning/potential_field.md) |

### Trajectory Generation

| Algorithm | Simulation | Docs |
|-----------|-----------|------|
| **Minimum-Snap** | ![MinSnap](simulations/min_snap/min_snap.gif) | [docs/trajectory/min_snap.md](docs/trajectory/min_snap.md) |
| **Polynomial** | ![Poly](simulations/polynomial_trajectory/polynomial_trajectory.gif) | [docs/trajectory/polynomial.md](docs/trajectory/polynomial.md) |

### Trajectory Tracking

| Algorithm | Simulation | Docs |
|-----------|-----------|------|
| **Feedback Linearisation** | ![FBL](simulations/feedback_linearisation/feedback_linearisation.gif) | [docs/tracking/feedback_linearisation.md](docs/tracking/feedback_linearisation.md) |
| **MPPI** | ![MPPI](simulations/mppi/mppi.gif) | [docs/tracking/mppi.md](docs/tracking/mppi.md) |

### Swarm Algorithms

| Algorithm | Simulation | Docs |
|-----------|-----------|------|
| **Reynolds Flocking** | ![Flock](simulations/reynolds_flocking/reynolds_flocking.gif) | [docs/swarm/reynolds_flocking.md](docs/swarm/reynolds_flocking.md) |
| **Consensus Formation** | ![Consensus](simulations/consensus_formation/consensus_formation.gif) | [docs/swarm/consensus_formation.md](docs/swarm/consensus_formation.md) |
| **Virtual Structure** | ![VS](simulations/virtual_structure/virtual_structure.gif) | [docs/swarm/virtual_structure.md](docs/swarm/virtual_structure.md) |
| **Leader-Follower** | ![LF](simulations/leader_follower/leader_follower.gif) | [docs/swarm/leader_follower.md](docs/swarm/leader_follower.md) |
| **Potential-Based Swarm** | ![PS](simulations/potential_swarm/potential_swarm.gif) | [docs/swarm/potential_swarm.md](docs/swarm/potential_swarm.md) |
| **Voronoi Coverage** | ![VC](simulations/voronoi_coverage/voronoi_coverage.gif) | [docs/swarm/voronoi_coverage.md](docs/swarm/voronoi_coverage.md) |

### Models & Infrastructure

| Component | Docs |
|-----------|------|
| 6DOF Quadrotor Dynamics | [docs/models/quadrotor.md](docs/models/quadrotor.md) |
| Motor Model | [docs/models/motor.md](docs/models/motor.md) |
| Control Allocation Mixer | [docs/models/mixer.md](docs/models/mixer.md) |

## Project Structure

```
src/quadrotor_sim/
├── models/              # Quadrotor dynamics, motors, mixer
├── control/             # PID, LQR, Geometric
├── estimation/          # Complementary, EKF, UKF, Particle
├── planning/            # A*, RRT*, Potential Field, Min-Snap, Polynomial
├── tracking/            # Feedback Linearisation, MPPI
├── swarm/               # Reynolds, Consensus, Virtual Structure, Leader-Follower, Potential, Coverage
└── visualization/       # SimAnimator for GIF generation

simulations/
├── pid_hover/           # Each algorithm has its own folder
│   ├── run.py           # Runnable simulation script
│   └── pid_hover.gif    # Generated animation
├── lqr_hover/
│   └── ...
└── ...

docs/                    # One doc per algorithm: theory, reference, API
tests/                   # Comprehensive unit tests
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) — Erwin Lejeune, 2026.
