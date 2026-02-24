<!-- Erwin Lejeune — 2026-02-23 -->
# Simulations

All simulations are self-contained Python modules under `src/uav_sim/simulations/`. Each produces a GIF animation and a JSON log.

## Categories

| Category | Simulations | Key Concepts |
|----------|------------|--------------|
| [Estimation](/simulations/estimation/ekf) | EKF, UKF, Complementary, Particle Filter, GPS-IMU Fusion | Sensor fusion, state estimation |
| [Path Tracking](/simulations/path-tracking/pid-hover) | PID Hover, LQR Hover, Flight Ops | Cascaded control, state machine |
| [Trajectory Planning](/simulations/trajectory-planning/min-snap) | Min-Snap, Polynomial, Quintic, Frenet | Smooth path generation |
| [Trajectory Tracking](/simulations/trajectory-tracking/feedback-linearisation) | Feedback Lin., NMPC, MPPI | Optimal control |
| [Perception](/simulations/perception/ekf-slam) | EKF-SLAM, Occupancy Mapping, Visual Servoing | Mapping, target following |
| [Sensors](/simulations/sensors/gimbal-tracking) | Gimbal Tracking, BBox Tracking | Camera control |
| [Swarm](/simulations/swarm/reynolds-flocking) | Reynolds, Voronoi, Leader-Follower, Consensus, Virtual Structure, Potential | Multi-agent coordination |
| [Environment](/simulations/environment/costmap-navigation) | Costmap Navigation | Obstacle avoidance |

## Running

```bash
python -m uav_sim.simulations.<category>.<name>
```
