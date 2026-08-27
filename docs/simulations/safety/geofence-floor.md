<!-- Erwin Lejeune — 2026-08-27 -->
# Geofence and Floor

## Problem Statement

The nominal command does the two things a geofence exists to prevent: flat
out at the boundary, then a dive at the ground. Two barriers refuse it.

## Model and Formulation

Each box face and the altitude floor is its own barrier on *position*, so the
input reaches them only through the second derivative. The filter therefore
begins decelerating well before the limit rather than clamping at it.

A clamp is not a safety guarantee. It is a report that safety was already
lost, and the unfiltered run shows what that looks like.

## Tuning and Failure Modes

- **Do not post-clip.** Clipping a solved command can push it back across a
  barrier the QP had just satisfied; actuator limits belong *in* the QP.
- The class-K gains set how early the deceleration starts. Too soft and the
  vehicle overshoots on a fast approach; too stiff and it refuses to
  approach the boundary at all.

## Evidence

| run | min wall margin | min floor margin |
|---|---|---|
| filtered | **+0.00 m** | **+0.00 m** |
| unfiltered | −13463 m | −4481 m |

![Geofence and Floor](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/safety/geofence_floor/geofence_floor.gif)

## References

- [Xiao and Belta, High Order Control Barrier Functions (2022)](https://doi.org/10.1109/TAC.2021.3105491)

## Related Algorithms

- [Position Exchange](/simulations/safety/position-exchange)
- [Obstacle Slalom](/simulations/safety/obstacle-slalom)
