<!-- Erwin Lejeune — 2026-08-27 -->
# Obstacle Slalom

## Problem Statement

The nominal controller flies dead straight at the goal and has never heard of
the obstacles. Everything keeping the vehicle out of them is the barrier —
which is the point: the safety argument does not depend on the controller
being any good.

## Model and Formulation

$h = \lVert p - c \rVert^2 - (r + \delta)^2$ for each obstacle, with relative
degree two under acceleration control, so the high-order condition applies.
Unlike the pairwise case the obstacle does not manoeuvre, so the whole
avoidance burden is the vehicle's and each constraint touches one block of
the decision vector.

## Tuning and Failure Modes

- **Offset the obstacles by less than a radius from the direct path.** At
  exactly one radius the straight line is tangent, and the unfiltered run
  looks almost safe by accident — the first version of this scene grazed at
  +0.001 m instead of penetrating.
- Clearance is a separate parameter from radius so the margin can be tuned
  without moving the obstacle.

## Evidence

| run | min clearance | goal error |
|---|---|---|
| filtered | **+1.189 m** | 0.00 m |
| unfiltered | **−1.997 m** | 0.00 m |

Required clearance 1.200 m; the filtered run sits 1 % inside it, which is
discretisation overshoot on a condition that holds in continuous time.

![Obstacle Slalom](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/safety/obstacle_slalom/obstacle_slalom.gif)

## References

- [Ames et al., CBF-based Quadratic Programs for Safety Critical Systems (2017)](https://doi.org/10.1109/TAC.2016.2638961)

## Related Algorithms

- [Position Exchange](/simulations/safety/position-exchange)
- [Potential Field 3D](/simulations/path-planning/potential-field-3d)
