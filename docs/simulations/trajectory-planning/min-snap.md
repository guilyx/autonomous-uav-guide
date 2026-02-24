<!-- Erwin Lejeune — 2026-02-24 -->
# Minimum-Snap Trajectory

## Problem Statement

Waypoint-only paths are not directly flyable because they ignore high-order dynamic smoothness.
Minimum-snap planning computes polynomial trajectories that reduce aggressive jerk/snap behavior and improve tracking performance.

## Model and Formulation

Each segment is represented by a polynomial:

$$
p(t) = \sum_{i=0}^{n} c_i t^i
$$

The optimization minimizes integrated snap:

$$
J = \int_{0}^{T}\left\|\frac{d^4p(t)}{dt^4}\right\|^2dt
$$

subject to waypoint and continuity constraints for position, velocity, acceleration, and jerk.

## Algorithm Procedure

1. Allocate segment times across waypoints.
2. Build quadratic cost matrix for snap objective.
3. Apply boundary and continuity equality constraints.
4. Solve constrained QP for polynomial coefficients.

## Tuning Guidance

- Time allocation dominates smoothness-quality trade-offs.
- Enforce corridor constraints for cluttered environments.
- Increase continuity order for aggressive maneuvers with tight tracking budgets.

## Failure Modes and Diagnostics

- Unrealistic segment times create numerically stiff trajectories.
- Sparse waypoints can violate obstacle-clearance assumptions.
- Overly smooth trajectories may become too conservative for time-critical tasks.

## Implementation and Execution

```bash
python -m uav_sim.simulations.trajectory_planning.min_snap
```

## Evidence

![Minimum Snap](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/min_snap/min_snap.gif)

## References

- [Mellinger and Kumar, Minimum Snap Trajectory Generation and Control for Quadrotors (2011)](https://doi.org/10.1109/ICRA.2011.5980409)
- [Richter, Bry, Roy, Polynomial Trajectory Planning for Aggressive Quadrotor Flight (2016)](https://doi.org/10.1177/0278364916631530)

## Related Algorithms

- [Polynomial Trajectory](/simulations/trajectory-planning/polynomial)
- [Feedback Linearisation](/simulations/trajectory-tracking/feedback-linearisation)
