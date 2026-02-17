# Autonomous UAV Guide

[![CI](https://github.com/guilyx/autonomous-uav-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/guilyx/autonomous-uav-guide/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

From-scratch Python implementations of algorithms for **autonomous UAVs**: multirotor, VTOL, and fixed-wing. Every algorithm comes with a runnable simulation, academic references, and a GIF preview.

> See [docs/architecture.md](docs/architecture.md) for the full package structure and design decisions.

## Quick Start

```bash
# Clone & install
git clone https://github.com/guilyx/autonomous-uav-guide.git
cd autonomous-uav-guide
uv sync --all-groups

# Run any simulation
uv run python simulations/path_tracking/pid_hover/run.py

# Run tests
uv run pytest

# Pre-commit
pre-commit install && pre-commit install --hook-type commit-msg
pre-commit run --all-files
```

---

## Vehicle Models

| Model | Docs | Preview |
|---|---|---|
| **Quadrotor** (6DOF + motor dynamics) | [docs](docs/vehicles/quadrotor.md) | <img src="simulations/path_tracking/pid_hover/pid_hover.gif" width="200"/> |
| **Fixed-Wing** (aerodynamic) | [docs](docs/vehicles/fixed_wing.md) | <img src="simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif" width="200"/> |
| **Tilt-Rotor VTOL** (hover ↔ cruise) | [docs](docs/vehicles/vtol.md) | <img src="simulations/vehicles/vtol_transition/vtol_transition.gif" width="200"/> |

## Path Tracking

| Algorithm | Docs | Preview |
|---|---|---|
| **Cascaded PID** | [docs](docs/path_tracking/pid.md) | <img src="simulations/path_tracking/pid_hover/pid_hover.gif" width="200"/> |
| **LQR** | [docs](docs/path_tracking/lqr.md) | <img src="simulations/path_tracking/lqr_hover/lqr_hover.gif" width="200"/> |
| **Geometric SO(3)** | [docs](docs/path_tracking/geometric.md) | <img src="simulations/path_tracking/geometric_control/geometric_control.gif" width="200"/> |
| **Waypoint Tracking** | [docs](docs/path_tracking/pid.md) | <img src="simulations/path_tracking/waypoint_tracking/waypoint_tracking.gif" width="200"/> |

## Trajectory Tracking

| Algorithm | Docs | Preview |
|---|---|---|
| **Feedback Linearisation** | [docs](docs/trajectory_tracking/feedback_linearisation.md) | <img src="simulations/trajectory_tracking/feedback_linearisation/feedback_linearisation.gif" width="200"/> |
| **MPPI** | [docs](docs/trajectory_tracking/mppi.md) | <img src="simulations/trajectory_tracking/mppi/mppi.gif" width="200"/> |

## Path Planning

| Algorithm | Docs | Preview |
|---|---|---|
| **3D A\*** | [docs](docs/path_planning/astar_3d.md) | <img src="simulations/path_planning/astar_3d/astar_3d.gif" width="200"/> |
| **RRT\*** | [docs](docs/path_planning/rrt_star.md) | <img src="simulations/path_planning/rrt_star_3d/rrt_star_3d.gif" width="200"/> |
| **Potential Field** | [docs](docs/path_planning/potential_field.md) | <img src="simulations/path_planning/potential_field_3d/potential_field_3d.gif" width="200"/> |

## Trajectory Planning

| Algorithm | Docs | Preview |
|---|---|---|
| **Minimum-Snap** | [docs](docs/trajectory_planning/min_snap.md) | <img src="simulations/trajectory_planning/min_snap/min_snap.gif" width="200"/> |
| **Polynomial** | [docs](docs/trajectory_planning/polynomial.md) | <img src="simulations/trajectory_planning/polynomial_trajectory/polynomial_trajectory.gif" width="200"/> |

## State Estimation

| Algorithm | Docs | Preview |
|---|---|---|
| **EKF** | [docs](docs/estimation/ekf.md) | <img src="simulations/estimation/ekf/ekf.gif" width="200"/> |
| **UKF** | [docs](docs/estimation/ukf.md) | <img src="simulations/estimation/ukf/ukf.gif" width="200"/> |
| **Particle Filter** | [docs](docs/estimation/particle_filter.md) | <img src="simulations/estimation/particle_filter/particle_filter.gif" width="200"/> |
| **Complementary Filter** | [docs](docs/estimation/complementary_filter.md) | <img src="simulations/estimation/complementary_filter/complementary_filter.gif" width="200"/> |

## Environment & Costmaps

| Feature | Docs | Preview |
|---|---|---|
| **Costmap Navigation** | [docs](docs/costmap/costmap.md) | <img src="simulations/environment/costmap_navigation/costmap_navigation.gif" width="200"/> |
| **World & Obstacles** | [docs](docs/environment/world.md) | — |

## Sensors & Perception

| Feature | Docs | Preview |
|---|---|---|
| **Lidar Mapping** | [docs](docs/perception/perception.md) | <img src="simulations/perception/lidar_mapping/lidar_mapping.gif" width="200"/> |
| **Sensor Suite** | [docs](docs/sensors/sensors.md) | — |

## Swarm Algorithms

| Algorithm | Docs | Preview |
|---|---|---|
| **Reynolds Flocking** | [docs](docs/swarm/reynolds_flocking.md) | <img src="simulations/swarm/reynolds_flocking/reynolds_flocking.gif" width="200"/> |
| **Consensus Formation** | [docs](docs/swarm/consensus_formation.md) | <img src="simulations/swarm/consensus_formation/consensus_formation.gif" width="200"/> |
| **Virtual Structure** | [docs](docs/swarm/virtual_structure.md) | <img src="simulations/swarm/virtual_structure/virtual_structure.gif" width="200"/> |
| **Leader-Follower** | [docs](docs/swarm/leader_follower.md) | <img src="simulations/swarm/leader_follower/leader_follower.gif" width="200"/> |
| **Potential Swarm** | [docs](docs/swarm/potential_swarm.md) | <img src="simulations/swarm/potential_swarm/potential_swarm.gif" width="200"/> |
| **Voronoi Coverage** | [docs](docs/swarm/voronoi_coverage.md) | <img src="simulations/swarm/voronoi_coverage/voronoi_coverage.gif" width="200"/> |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must pass `pre-commit run --all-files` and `uv run pytest`.

## License

MIT — see [LICENSE](LICENSE).
