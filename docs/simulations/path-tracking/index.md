<!-- Erwin Lejeune — 2026-02-24 -->
# Path Tracking

Path tracking converts geometric references into feasible attitude and thrust commands.
This chapter spans linear and nonlinear feedback structures used for hover and mission-level behavior.

## Core Questions

- How should positional and attitude loops be decoupled?
- Which error-state definitions improve transient response?
- How do saturation and actuator limits affect stability margins?

## Algorithms

- [PID Hover](/simulations/path-tracking/pid-hover)
- [LQR Hover](/simulations/path-tracking/lqr-hover)
- [LQR Path Tracking](/simulations/path-tracking/lqr-tracking)
- [MPC Tracking](/simulations/path-tracking/mpc-tracking)
- [Geometric Control on SE(3)](/simulations/path-tracking/geometric-control)
- [Pure Pursuit 3D](/simulations/path-tracking/pure-pursuit)
- [Fixed-Wing Mission Navigation](/simulations/path-tracking/fixed-wing-mission)
- [Path Smoothing](/simulations/path-tracking/path-smoothing)
- [Flight Ops Demo](/simulations/path-tracking/flight-ops-demo)

## The One Thing That Separates Hovering From Tracking

Every controller in this chapter can hold a point. What distinguishes
them on a *moving* reference is whether they are given its derivatives.

A pure feedback loop has to manufacture the reference's own acceleration
out of tracking error, so it settles at whatever error does the job —
error proportional to how hard the trajectory is. Feeding the reference
acceleration forward removes that term from the error dynamics entirely.
On the atlas figure-8 the difference is an order of magnitude for every
controller here:

| Controller | Feedback only | With feed-forward |
|---|---|---|
| LQR | 0.158 m | 0.017 m |
| Geometric SO(3) | 0.388 m | 0.023 m |
| MPC (preview) | 0.395 m | 0.032 m |
| Feedback linearisation | 0.234 m | 0.014 m |

The symptom to recognise: **tracking error that scales with reference
speed, and an overshoot on the trajectory's fastest axis.** That is not a
gain that needs raising.

## Prerequisites

- State-space control basics
- Hover linearization around trim
- Attitude representation in SO(3)/Euler form
