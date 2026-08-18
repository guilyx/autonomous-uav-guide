<!-- Erwin Lejeune — 2026-02-24 -->
# Algorithm atlas

Every algorithm in the library has a runnable simulation that renders a
three-panel animation, writes a JSON log, and cites its source.

```bash
flybots list                 # browse the catalogue
flybots info astar_3d        # references and usage for one
flybots run astar_3d         # render it
```

Or as a module: `python -m uav_sim.simulations.<category>.<name>`.

<StatBand :items="[
  { value: '42', label: 'simulations' },
  { value: '10', label: 'domains' },
  { value: '100%', label: 'with references' },
]" />

::: tip Reading order
Estimation → control and tracking → planning → perception → swarm. Each
builds on the state representation the previous one establishes.
:::

## Vehicles

| Algorithm | Article | Preview |
|---|---|---|
| Quadrotor dynamics | [Open](/simulations/vehicles/quadrotor-dynamics) | ![quadrotor](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif) |
| Fixed-wing flight | [Open](/simulations/vehicles/fixed-wing-flight) | ![fixed wing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif) |
| VTOL transition | [Open](/simulations/vehicles/vtol-transition) | ![vtol](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/vtol_transition/vtol_transition.gif) |

See also the model documentation: [quadrotor](/vehicles/quadrotor),
[fixed-wing](/vehicles/fixed-wing), [VTOL](/vehicles/vtol).

## Estimation

| Algorithm | Article | Preview |
|---|---|---|
| Complementary filter | [Open](/simulations/estimation/complementary-filter) | ![complementary](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/complementary_filter/complementary_filter.gif) |
| Extended Kalman filter | [Open](/simulations/estimation/ekf) | ![ekf](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ekf/ekf.gif) |
| Unscented Kalman filter | [Open](/simulations/estimation/ukf) | ![ukf](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ukf/ukf.gif) |
| GPS-IMU fusion | [Open](/simulations/estimation/gps-imu-fusion) | ![gps imu](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/gps_imu_fusion/gps_imu_fusion.gif) |
| Particle filter | [Open](/simulations/estimation/particle-filter) | ![particle](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/particle_filter/particle_filter.gif) |

## Control and path tracking

| Algorithm | Article | Preview |
|---|---|---|
| PID hover | [Open](/simulations/path-tracking/pid-hover) | ![pid](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pid_hover/pid_hover.gif) |
| LQR hover | [Open](/simulations/path-tracking/lqr-hover) | ![lqr](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/lqr_hover/lqr_hover.gif) |
| LQR path tracking | [Open](/simulations/path-tracking/lqr-tracking) | ![lqr tracking](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/lqr_tracking/lqr_tracking.gif) |
| MPC tracking | [Open](/simulations/path-tracking/mpc-tracking) | ![mpc](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/mpc_tracking/mpc_tracking.gif) |
| Geometric SO(3) | [Open](/simulations/path-tracking/geometric-control) | ![geometric](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/geometric_control/geometric_control.gif) |
| Pure pursuit 3D | [Open](/simulations/path-tracking/pure-pursuit) | ![pure pursuit](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/pure_pursuit/pure_pursuit.gif) |
| Path smoothing | [Open](/simulations/path-tracking/path-smoothing) | ![path smoothing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/path_smoothing_demo/path_smoothing_demo.gif) |
| Flight operations sequence | [Open](/simulations/path-tracking/flight-ops-demo) | ![flight ops](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/flight_ops_demo/flight_ops_demo.gif) |

## Path planning

| Algorithm | Article | Preview |
|---|---|---|
| A* 3D | [Open](/simulations/path-planning/astar-3d) | ![astar](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/astar_3d/astar_3d.gif) |
| RRT* 3D | [Open](/simulations/path-planning/rrt-star-3d) | ![rrt star](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/rrt_star_3d/rrt_star_3d.gif) |
| PRM 3D | [Open](/simulations/path-planning/prm-3d) | ![prm](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/prm_3d/prm_3d.gif) |
| Potential field 3D | [Open](/simulations/path-planning/potential-field-3d) | ![potential](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/potential_field_3d/potential_field_3d.gif) |
| Coverage planning | [Open](/simulations/path-planning/coverage-planning) | ![coverage](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/coverage_planning/coverage_planning.gif) |

## Trajectory planning

| Algorithm | Article | Preview |
|---|---|---|
| Minimum snap | [Open](/simulations/trajectory-planning/min-snap) | ![min snap](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/min_snap/min_snap.gif) |
| Polynomial trajectory | [Open](/simulations/trajectory-planning/polynomial) | ![polynomial](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/polynomial_trajectory/polynomial_trajectory.gif) |
| Quintic polynomial | [Open](/simulations/trajectory-planning/quintic) | ![quintic](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/quintic_polynomial_demo/quintic_polynomial_demo.gif) |
| Frenet optimal | [Open](/simulations/trajectory-planning/frenet-optimal) | ![frenet](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/frenet_optimal/frenet_optimal.gif) |

## Trajectory tracking

| Algorithm | Article | Preview |
|---|---|---|
| Feedback linearisation | [Open](/simulations/trajectory-tracking/feedback-linearisation) | ![feedback lin](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/feedback_linearisation/feedback_linearisation.gif) |
| NMPC | [Open](/simulations/trajectory-tracking/nmpc) | ![nmpc](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/nmpc/nmpc.gif) |
| MPPI | [Open](/simulations/trajectory-tracking/mppi) | ![mppi](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/mppi/mppi.gif) |

## Perception

| Feature | Article | Preview |
|---|---|---|
| EKF-SLAM | [Open](/simulations/perception/ekf-slam) | ![ekf slam](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/ekf_slam/ekf_slam.gif) |
| Occupancy mapping | [Open](/simulations/perception/occupancy-mapping) | ![occupancy](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/occupancy_mapping/occupancy_mapping.gif) |
| Visual servoing | [Open](/simulations/perception/visual-servoing) | ![visual servoing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/visual_servoing/visual_servoing.gif) |
| Sensor suite | [Open](/simulations/perception/sensor-suite) | ![sensor suite](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/sensor_suite_demo/sensor_suite_demo.gif) |

## Sensors

| Feature | Article | Preview |
|---|---|---|
| Gimbal FOV tracking | [Open](/simulations/sensors/gimbal-tracking) | ![gimbal](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_tracking/gimbal_tracking.gif) |
| Gimbal bounding-box tracking | [Open](/simulations/sensors/gimbal-bbox-tracking) | ![gimbal bbox](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_bbox_tracking/gimbal_bbox_tracking.gif) |

## Environment and costmaps

| Feature | Article | Preview |
|---|---|---|
| Dynamic costmap navigation | [Open](/simulations/environment/costmap-navigation) | ![costmap](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/environment/costmap_navigation/costmap_navigation.gif) |

## Swarm

| Algorithm | Article | Preview |
|---|---|---|
| Reynolds flocking | [Open](/simulations/swarm/reynolds-flocking) | ![reynolds](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/reynolds_flocking/reynolds_flocking.gif) |
| Consensus formation | [Open](/simulations/swarm/consensus-formation) | ![consensus](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/consensus_formation/consensus_formation.gif) |
| Virtual structure | [Open](/simulations/swarm/virtual-structure) | ![virtual structure](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/virtual_structure/virtual_structure.gif) |
| Leader-follower | [Open](/simulations/swarm/leader-follower) | ![leader follower](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/leader_follower/leader_follower.gif) |
| Potential swarm | [Open](/simulations/swarm/potential-swarm) | ![potential swarm](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/potential_swarm/potential_swarm.gif) |
| Voronoi coverage | [Open](/simulations/swarm/voronoi-coverage) | ![voronoi](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/voronoi_coverage/voronoi_coverage.gif) |

## Reinforcement learning

Not simulations but environments — see
[Reinforcement learning](/learning/) for the six tasks and the trainer.

## Notes on the previews

Preview GIFs are stored in Git LFS and served through
`media.githubusercontent.com`. Regenerate any of them locally with
`flybots run <name>`; they are written next to the simulation's `run.py`.
