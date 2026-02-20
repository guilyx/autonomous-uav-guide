# Autonomous UAV Guide

[![CI](https://github.com/guilyx/autonomous-uav-guide/actions/workflows/ci.yml/badge.svg)](https://github.com/guilyx/autonomous-uav-guide/actions/workflows/ci.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/db279b02fee84675a89f6f4892f23d30)](https://app.codacy.com/gh/guilyx/autonomous-uav-guide/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

From-scratch Python implementations of algorithms for **autonomous UAVs**: multirotor, VTOL, and fixed-wing. Every algorithm comes with a runnable simulation, academic references, and a GIF preview.

## Architecture

```
uav_sim/
├── vehicles/        # Quadrotor (6DOF), Fixed-Wing, VTOL + vehicle presets
│   └── footprint.py # Circular/Rectangular footprints, swarm envelopes
├── control/         # Rate → Attitude → Velocity → Position (cascaded PID)
│   └── state_machine.py  # ARM → TAKEOFF → HOVER → TRACKING → LAND
├── sensors/         # GPS, IMU, Lidar2D/3D, Camera, Gimbal, RangeFinder
│   └── gimbal_controller.py  # PointTracker, BBoxTracker
├── estimation/      # EKF, UKF, Complementary, Particle Filter
├── perception/      # Occupancy mapping, obstacle detection, visual servoing
├── path_planning/   # A*, RRT*, PRM, Potential Field, Coverage Planning
├── path_tracking/   # PID, LQR, MPC, Pure Pursuit, Geometric SO(3)
├── trajectory_planning/  # Min-Snap, Polynomial, Quintic, Frenet Optimal
├── trajectory_tracking/  # Feedback Lin., MPPI, NMPC
├── costmap/         # Occupancy grid, Inflation, Social, Footprint layers
├── environment/     # World, obstacles, buildings, env presets (city/indoor/field)
├── swarm/           # Reynolds, Consensus, Virtual Structure, Leader-Follower
├── visualization/   # 3-panel viz, data panels, vehicle artists, sensor viz
└── simulations/     # 41 runnable demos (python -m uav_sim.simulations.*)
```

## Quick Start

```bash
git clone https://github.com/guilyx/autonomous-uav-guide.git
cd autonomous-uav-guide
uv sync --all-groups

# Run any simulation as a module
python -m uav_sim.simulations.path_tracking.pid_hover

# Run tests
uv run pytest

# Pre-commit
pre-commit install && pre-commit install --hook-type commit-msg
pre-commit run --all-files
```

### Vehicle Presets

```python
from uav_sim.vehicles import VehiclePreset, create_quadrotor

quad = create_quadrotor(VehiclePreset.CRAZYFLIE)
quad = create_quadrotor(VehiclePreset.DJI_MINI)
quad = create_quadrotor(VehiclePreset.RACING_250)
quad = create_quadrotor(VehiclePreset.DJI_MATRICE)
```

### Environment Presets

```python
from uav_sim.environment import create_environment, EnvironmentPreset

world, obs = create_environment(EnvironmentPreset.CITY)     # 50m urban
world, obs = create_environment(EnvironmentPreset.INDOOR)   # 10m room
world, obs = create_environment(EnvironmentPreset.OPEN_FIELD)  # 60m clear
```

---

## Vehicle Models

| Model | Preview |
|---|---|
| **Quadrotor** (6DOF + motor dynamics) | <img src="src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif" width="560"/> |
| **Fixed-Wing** (aerodynamic) | <img src="src/uav_sim/simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif" width="560"/> |
| **Tilt-Rotor VTOL** (hover to cruise) | <img src="src/uav_sim/simulations/vehicles/vtol_transition/vtol_transition.gif" width="560"/> |

## Path Tracking

| Algorithm | Preview |
|---|---|
| **Cascaded PID Hover** | <img src="src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif" width="560"/> |
| **LQR Hover** | <img src="src/uav_sim/simulations/path_tracking/lqr_hover/lqr_hover.gif" width="560"/> |
| **Pure Pursuit 3D** | <img src="src/uav_sim/simulations/path_tracking/pure_pursuit/pure_pursuit.gif" width="560"/> |
| **Geometric SO(3)** | <img src="src/uav_sim/simulations/path_tracking/geometric_control/geometric_control.gif" width="560"/> |
| **LQR Path Tracking** | <img src="src/uav_sim/simulations/path_tracking/lqr_tracking/lqr_tracking.gif" width="560"/> |
| **MPC Tracking** | <img src="src/uav_sim/simulations/path_tracking/mpc_tracking/mpc_tracking.gif" width="560"/> |
| **Path Smoothing Demo** | <img src="src/uav_sim/simulations/path_tracking/path_smoothing_demo/path_smoothing_demo.gif" width="560"/> |
| **Flight Ops Demo** | <img src="src/uav_sim/simulations/path_tracking/flight_ops_demo/flight_ops_demo.gif" width="560"/> |

## Trajectory Tracking

| Algorithm | Preview |
|---|---|
| **Feedback Linearisation** | <img src="src/uav_sim/simulations/trajectory_tracking/feedback_linearisation/feedback_linearisation.gif" width="560"/> |
| **MPPI** | <img src="src/uav_sim/simulations/trajectory_tracking/mppi/mppi.gif" width="560"/> |
| **NMPC** | <img src="src/uav_sim/simulations/trajectory_tracking/nmpc/nmpc.gif" width="560"/> |

## Path Planning

| Algorithm | Preview |
|---|---|
| **3D A\*** | <img src="src/uav_sim/simulations/path_planning/astar_3d/astar_3d.gif" width="560"/> |
| **RRT\*** | <img src="src/uav_sim/simulations/path_planning/rrt_star_3d/rrt_star_3d.gif" width="560"/> |
| **PRM 3D** | <img src="src/uav_sim/simulations/path_planning/prm_3d/prm_3d.gif" width="560"/> |
| **Potential Field** | <img src="src/uav_sim/simulations/path_planning/potential_field_3d/potential_field_3d.gif" width="560"/> |
| **Coverage Planning** | <img src="src/uav_sim/simulations/path_planning/coverage_planning/coverage_planning.gif" width="560"/> |

## Trajectory Planning

| Algorithm | Preview |
|---|---|
| **Minimum-Snap** | <img src="src/uav_sim/simulations/trajectory_planning/min_snap/min_snap.gif" width="560"/> |
| **Polynomial Trajectory** | <img src="src/uav_sim/simulations/trajectory_planning/polynomial_trajectory/polynomial_trajectory.gif" width="560"/> |
| **Quintic Polynomial** | <img src="src/uav_sim/simulations/trajectory_planning/quintic_polynomial_demo/quintic_polynomial_demo.gif" width="560"/> |
| **Frenet Optimal** | <img src="src/uav_sim/simulations/trajectory_planning/frenet_optimal/frenet_optimal.gif" width="560"/> |

## State Estimation

| Algorithm | Preview |
|---|---|
| **EKF** | <img src="src/uav_sim/simulations/estimation/ekf/ekf.gif" width="560"/> |
| **UKF** | <img src="src/uav_sim/simulations/estimation/ukf/ukf.gif" width="560"/> |
| **Particle Filter** | <img src="src/uav_sim/simulations/estimation/particle_filter/particle_filter.gif" width="560"/> |
| **Complementary Filter** | <img src="src/uav_sim/simulations/estimation/complementary_filter/complementary_filter.gif" width="560"/> |
| **GPS/IMU Fusion** | <img src="src/uav_sim/simulations/estimation/gps_imu_fusion/gps_imu_fusion.gif" width="560"/> |

## Perception

| Feature | Preview |
|---|---|
| **EKF-SLAM** | <img src="src/uav_sim/simulations/perception/ekf_slam/ekf_slam.gif" width="560"/> |
| **Occupancy Mapping** | <img src="src/uav_sim/simulations/perception/occupancy_mapping/occupancy_mapping.gif" width="560"/> |
| **Sensor Suite Demo** | <img src="src/uav_sim/simulations/perception/sensor_suite_demo/sensor_suite_demo.gif" width="560"/> |
| **Visual Servoing** | <img src="src/uav_sim/simulations/perception/visual_servoing/visual_servoing.gif" width="560"/> |

## Sensors

| Feature | Preview |
|---|---|
| **Gimbal FOV Tracking** | <img src="src/uav_sim/simulations/sensors/gimbal_tracking/gimbal_tracking.gif" width="560"/> |
| **Gimbal BBox Tracking** | <img src="src/uav_sim/simulations/sensors/gimbal_bbox_tracking/gimbal_bbox_tracking.gif" width="560"/> |

## Environment & Costmaps

| Feature | Preview |
|---|---|
| **Dynamic Costmap Navigation** | <img src="src/uav_sim/simulations/environment/costmap_navigation/costmap_navigation.gif" width="560"/> |

## Swarm Algorithms

| Algorithm | Preview |
|---|---|
| **Reynolds Flocking** | <img src="src/uav_sim/simulations/swarm/reynolds_flocking/reynolds_flocking.gif" width="560"/> |
| **Consensus Formation** | <img src="src/uav_sim/simulations/swarm/consensus_formation/consensus_formation.gif" width="560"/> |
| **Virtual Structure** | <img src="src/uav_sim/simulations/swarm/virtual_structure/virtual_structure.gif" width="560"/> |
| **Leader-Follower** | <img src="src/uav_sim/simulations/swarm/leader_follower/leader_follower.gif" width="560"/> |
| **Potential Swarm** | <img src="src/uav_sim/simulations/swarm/potential_swarm/potential_swarm.gif" width="560"/> |
| **Voronoi Coverage** | <img src="src/uav_sim/simulations/swarm/voronoi_coverage/voronoi_coverage.gif" width="560"/> |

---

## Contributing [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/guilyx/autonomous-uav-guide/issues)

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions must pass `pre-commit run --all-files` and `uv run pytest`.

## License

MIT — see [LICENSE](LICENSE).
