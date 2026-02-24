<!-- Erwin Lejeune — 2026-02-23 -->
# Simulation Catalog

All simulation modules live in `src/uav_sim/simulations/`.
Each run generates:

- a GIF animation (`*.gif`)
- a structured time-series log (`*_log.json`)

GitHub GIF base URL used below:
`https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/`

## Vehicles

| Simulation | Preview |
|---|---|
| Quadrotor PID Hover | ![pid hover](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif) |
| Fixed-Wing Flight | ![fixed wing](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif) |
| VTOL Transition | ![vtol](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/vtol_transition/vtol_transition.gif) |

## Path Tracking

| Simulation | Preview |
|---|---|
| PID Hover | ![pid](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif) |
| LQR Hover | ![lqr](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/lqr_hover/lqr_hover.gif) |
| Pure Pursuit 3D | ![pure pursuit](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pure_pursuit/pure_pursuit.gif) |
| Geometric SO(3) | ![geometric](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/geometric_control/geometric_control.gif) |
| LQR Tracking | ![lqr tracking](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/lqr_tracking/lqr_tracking.gif) |
| MPC Tracking | ![mpc tracking](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/mpc_tracking/mpc_tracking.gif) |
| Path Smoothing | ![smoothing](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/path_smoothing_demo/path_smoothing_demo.gif) |
| Flight Ops Demo | ![flight ops](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/flight_ops_demo/flight_ops_demo.gif) |

## Trajectory Tracking

| Simulation | Preview |
|---|---|
| Feedback Linearisation | ![feedback lin](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/feedback_linearisation/feedback_linearisation.gif) |
| MPPI | ![mppi](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/mppi/mppi.gif) |
| NMPC | ![nmpc](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/nmpc/nmpc.gif) |

## Path Planning

| Simulation | Preview |
|---|---|
| A* 3D | ![astar](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/astar_3d/astar_3d.gif) |
| RRT* 3D | ![rrt star](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/rrt_star_3d/rrt_star_3d.gif) |
| PRM 3D | ![prm](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/prm_3d/prm_3d.gif) |
| Potential Field 3D | ![potential](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/potential_field_3d/potential_field_3d.gif) |
| Coverage Planning | ![coverage](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/coverage_planning/coverage_planning.gif) |

## Trajectory Planning

| Simulation | Preview |
|---|---|
| Minimum Snap | ![min snap](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/min_snap/min_snap.gif) |
| Polynomial Trajectory | ![polynomial](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/polynomial_trajectory/polynomial_trajectory.gif) |
| Quintic Polynomial | ![quintic](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/quintic_polynomial_demo/quintic_polynomial_demo.gif) |
| Frenet Optimal | ![frenet](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/frenet_optimal/frenet_optimal.gif) |

## Estimation

| Simulation | Preview |
|---|---|
| EKF | ![ekf](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ekf/ekf.gif) |
| UKF | ![ukf](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ukf/ukf.gif) |
| Particle Filter | ![particle filter](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/particle_filter/particle_filter.gif) |
| Complementary Filter | ![complementary](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/complementary_filter/complementary_filter.gif) |
| GPS / IMU Fusion | ![gps imu](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/gps_imu_fusion/gps_imu_fusion.gif) |

## Perception

| Simulation | Preview |
|---|---|
| EKF-SLAM | ![ekf slam](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/ekf_slam/ekf_slam.gif) |
| Occupancy Mapping | ![occupancy](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/occupancy_mapping/occupancy_mapping.gif) |
| Sensor Suite Demo | ![sensor suite](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/sensor_suite_demo/sensor_suite_demo.gif) |
| Visual Servoing | ![visual servoing](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/visual_servoing/visual_servoing.gif) |

## Sensors

| Simulation | Preview |
|---|---|
| Gimbal FOV Tracking | ![gimbal tracking](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_tracking/gimbal_tracking.gif) |
| Gimbal BBox Tracking | ![gimbal bbox](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_bbox_tracking/gimbal_bbox_tracking.gif) |

## Environment

| Simulation | Preview |
|---|---|
| Dynamic Costmap Navigation | ![costmap nav](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/environment/costmap_navigation/costmap_navigation.gif) |

## Swarm

| Simulation | Preview |
|---|---|
| Reynolds Flocking | ![reynolds](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/reynolds_flocking/reynolds_flocking.gif) |
| Consensus Formation | ![consensus](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/consensus_formation/consensus_formation.gif) |
| Virtual Structure | ![virtual structure](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/virtual_structure/virtual_structure.gif) |
| Leader-Follower | ![leader follower](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/leader_follower/leader_follower.gif) |
| Potential Swarm | ![potential swarm](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/potential_swarm/potential_swarm.gif) |
| Voronoi Coverage | ![voronoi](https://raw.githubusercontent.com/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/voronoi_coverage/voronoi_coverage.gif) |

## Run Any Simulation

```bash
python -m uav_sim.simulations.<category>.<name>
```
