<!-- Erwin Lejeune — 2026-02-24 -->
# Algorithm Atlas

All algorithm implementations live under `src/uav_sim/simulations/` and are documented here as technical articles.
Each article links mathematical foundations to implementation behavior with reproducible GIF evidence.

## Conventions

- Media base URL: `https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/`
- Module execution pattern: `python -m uav_sim.simulations.<chapter>.<algorithm>`
- Recommended reading order: estimation -> control and tracking -> planning -> perception and swarm

## Estimation

| Algorithm | Article | Evidence |
|---|---|---|
| Complementary Filter | [Open](/simulations/estimation/complementary-filter) | ![complementary](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/complementary_filter/complementary_filter.gif) |
| Extended Kalman Filter | [Open](/simulations/estimation/ekf) | ![ekf](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ekf/ekf.gif) |
| Unscented Kalman Filter | [Open](/simulations/estimation/ukf) | ![ukf](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ukf/ukf.gif) |
| GPS-IMU Fusion | [Open](/simulations/estimation/gps-imu-fusion) | ![gps imu](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/gps_imu_fusion/gps_imu_fusion.gif) |
| Particle Filter | [Open](/simulations/estimation/particle-filter) | ![particle](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/particle_filter/particle_filter.gif) |

## Control and Path Tracking

| Algorithm | Article | Evidence |
|---|---|---|
| PID Hover | [Open](/simulations/path-tracking/pid-hover) | ![pid](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif) |
| LQR Hover | [Open](/simulations/path-tracking/lqr-hover) | ![lqr](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/lqr_hover/lqr_hover.gif) |
| Flight Operations Sequence | [Open](/simulations/path-tracking/flight-ops-demo) | ![flight ops](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/flight_ops_demo/flight_ops_demo.gif) |

## Path Planning

| Algorithm | Article | Evidence |
|---|---|---|
| A* 3D | [Open](/simulations/path-planning/astar-3d) | ![astar](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/astar_3d/astar_3d.gif) |
| RRT* 3D | [Open](/simulations/path-planning/rrt-star-3d) | ![rrt star](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/rrt_star_3d/rrt_star_3d.gif) |
| PRM 3D | [Open](/simulations/path-planning/prm-3d) | ![prm](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/prm_3d/prm_3d.gif) |
| Potential Field 3D | [Open](/simulations/path-planning/potential-field-3d) | ![potential](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/potential_field_3d/potential_field_3d.gif) |
| Coverage Planning | [Open](/simulations/path-planning/coverage-planning) | ![coverage](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/coverage_planning/coverage_planning.gif) |

## Trajectory Planning

| Algorithm | Article | Evidence |
|---|---|---|
| Minimum Snap | [Open](/simulations/trajectory-planning/min-snap) | ![min snap](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/min_snap/min_snap.gif) |
| Polynomial Trajectory | [Open](/simulations/trajectory-planning/polynomial) | ![polynomial](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/polynomial_trajectory/polynomial_trajectory.gif) |
| Quintic Polynomial | [Open](/simulations/trajectory-planning/quintic) | ![quintic](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/quintic_polynomial_demo/quintic_polynomial_demo.gif) |
| Frenet Optimal | [Open](/simulations/trajectory-planning/frenet-optimal) | ![frenet](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/frenet_optimal/frenet_optimal.gif) |

## Trajectory Tracking

| Algorithm | Article | Evidence |
|---|---|---|
| Feedback Linearisation | [Open](/simulations/trajectory-tracking/feedback-linearisation) | ![feedback lin](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/feedback_linearisation/feedback_linearisation.gif) |
| MPPI | [Open](/simulations/trajectory-tracking/mppi) | ![mppi](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/mppi/mppi.gif) |
| NMPC | [Open](/simulations/trajectory-tracking/nmpc) | ![nmpc](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/nmpc/nmpc.gif) |

## Perception and Sensing

| Algorithm | Article | Evidence |
|---|---|---|
| EKF-SLAM | [Open](/simulations/perception/ekf-slam) | ![ekf slam](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/ekf_slam/ekf_slam.gif) |
| Occupancy Mapping | [Open](/simulations/perception/occupancy-mapping) | ![occupancy](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/occupancy_mapping/occupancy_mapping.gif) |
| Sensor Suite Fusion | [Open](/simulations/perception/sensor-suite) | ![sensor suite](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/sensor_suite_demo/sensor_suite_demo.gif) |
| Visual Servoing | [Open](/simulations/perception/visual-servoing) | ![visual servoing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/visual_servoing/visual_servoing.gif) |
| Gimbal Tracking | [Open](/simulations/sensors/gimbal-tracking) | ![gimbal tracking](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_tracking/gimbal_tracking.gif) |
| Gimbal BBox Tracking | [Open](/simulations/sensors/gimbal-bbox-tracking) | ![gimbal bbox](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_bbox_tracking/gimbal_bbox_tracking.gif) |

## Environment

| Algorithm | Article | Evidence |
|---|---|---|
| Dynamic Costmap Navigation | [Open](/simulations/environment/costmap-navigation) | ![costmap nav](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/environment/costmap_navigation/costmap_navigation.gif) |

## Swarm

| Algorithm | Article | Evidence |
|---|---|---|
| Reynolds Flocking | [Open](/simulations/swarm/reynolds-flocking) | ![reynolds](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/reynolds_flocking/reynolds_flocking.gif) |
| Consensus Formation | [Open](/simulations/swarm/consensus-formation) | ![consensus](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/consensus_formation/consensus_formation.gif) |
| Virtual Structure | [Open](/simulations/swarm/virtual-structure) | ![virtual structure](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/virtual_structure/virtual_structure.gif) |
| Leader-Follower | [Open](/simulations/swarm/leader-follower) | ![leader follower](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/leader_follower/leader_follower.gif) |
| Potential Swarm | [Open](/simulations/swarm/potential-swarm) | ![potential swarm](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/potential_swarm/potential_swarm.gif) |
| Voronoi Coverage | [Open](/simulations/swarm/voronoi-coverage) | ![voronoi](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/voronoi_coverage/voronoi_coverage.gif) |

## Vehicles and Dynamics

| Algorithm | Article | Evidence |
|---|---|---|
| Quadrotor Dynamics | [Open](/simulations/vehicles/quadrotor-dynamics) | ![quadrotor](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif) |
| Fixed-Wing Flight | [Open](/simulations/vehicles/fixed-wing-flight) | ![fixed wing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif) |
| VTOL Transition | [Open](/simulations/vehicles/vtol-transition) | ![vtol](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/vtol_transition/vtol_transition.gif) |
