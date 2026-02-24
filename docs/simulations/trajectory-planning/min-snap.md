<!-- Erwin Lejeune — 2026-02-23 -->
# Minimum-Snap Trajectory

## Algorithm

Generates smooth polynomial trajectories that minimise the snap (4th derivative of position), producing dynamically feasible paths through waypoints.

**Reference:** D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA, 2011.

## Run

```bash
python -m uav_sim.simulations.trajectory_planning.min_snap
```
