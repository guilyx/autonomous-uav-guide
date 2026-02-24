<!-- Erwin Lejeune — 2026-02-24 -->
# LQR Hover

## Problem Statement

LQR hover control optimizes stabilization around a hover equilibrium by balancing state error and control effort.
It gives a principled gain matrix and clean performance trade-offs through weighting matrices.

## Model and Formulation

Linearized dynamics around hover:

$$
\dot{x} = Ax + Bu
$$

Objective:

$$
J = \int_0^\infty \left(x^\top Qx + u^\top Ru\right)dt
$$

Optimal policy:

$$
u = -Kx,\quad K = R^{-1}B^\top P
$$

where `P` solves the continuous-time algebraic Riccati equation.

## Algorithm Procedure

1. Linearize UAV dynamics around trim hover.
2. Choose `Q` and `R` according to tracking vs effort priorities.
3. Solve Riccati equation for `P`, then compute feedback gain `K`.
4. Apply full-state feedback in the hover envelope.

## Tuning Guidance

- Increase position-state weights in `Q` for tighter hover.
- Increase `R` to smooth actuation and reduce aggressive commands.
- Re-linearize if operating point drifts far from hover assumptions.

## Failure Modes and Diagnostics

- Performance degrades in strongly nonlinear/aggressive regimes.
- State-estimation latency can destabilize high-gain solutions.
- Poorly scaled units in `Q`/`R` create unintuitive controller behavior.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.lqr_hover
```

## Evidence

![LQR Hover](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/lqr_hover/lqr_hover.gif)

## References

- [Anderson and Moore, Optimal Control: Linear Quadratic Methods](https://books.google.com/books?id=iYMqAQAAMAAJ)
- [Bertsekas, Dynamic Programming and Optimal Control](https://www.mit.edu/~dimitrib/dpbook.html)

## Related Algorithms

- [PID Hover](/simulations/path-tracking/pid-hover)
- [NMPC](/simulations/trajectory-tracking/nmpc)
