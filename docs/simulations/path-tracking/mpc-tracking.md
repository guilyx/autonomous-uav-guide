<!-- Erwin Lejeune — 2026-08-13 -->
# MPC Path Tracking

## Problem Statement

Linear MPC uses the same hover-linearised model as [LQR](/simulations/path-tracking/lqr-tracking), but re-solves a finite-horizon optimal control problem every step and applies only the first input.
That buys two things LQR cannot offer: explicit input constraints, and **preview** of where the reference is going.

## Model and Formulation

At each control step:

$$
\min_{u_{0:N-1}} \sum_{k=0}^{N-1}\left(\lVert x_k - x_k^{ref}\rVert_Q^2 + \lVert u_k\rVert_R^2\right) + \lVert x_N - x_N^{ref}\rVert_{Q_f}^2
$$

subject to `x_{k+1} = A_d x_k + B_d u_k` and `u_min ≤ u_k ≤ u_max`.

## Preview Is the Whole Point

Note the `x_k^{ref}` — indexed by `k`, one reference per horizon step.
Holding a single reference point across the horizon asks the plan to come
to rest where the trajectory happens to be *right now*, and reduces MPC
to an LQR with a slower solver and a worse constant lag. Sampling the
reference forward is what makes it a different controller:

```python
preview = [reference(t + k * ctrl_dt) for k in range(horizon + 1)]
wrench = mpc.compute(
    state,
    np.array([p[0] for p in preview]),
    target_vel=np.array([p[1] for p in preview]),
)
```

On the atlas figure-8: **0.395 m** mean error held, **0.032 m** with preview.

## The Terminal Cost Has to Match the Discretisation

`Q_f` approximates the cost-to-go beyond the horizon, and it must come
from the **discrete** algebraic Riccati equation when the running cost is
summed per step rather than integrated. A continuous-time solution
under-weights the terminal state by a factor of `1/dt` — at `dt = 0.05`
that is 20× — and the horizon effectively ends in mid-air.

## Algorithm Procedure

1. Discretise the hover linearisation once, offline; solve the discrete ARE for `Q_f`.
2. Sample the reference across the horizon.
3. Warm-start from the previous solution, shifted by one step.
4. Solve the bounded QP, apply `u_0`, discard the rest.

## Tuning Guidance

- Longer horizons buy more preview and cost solve time. What matters is how far ahead the horizon *reaches* in seconds, not how many knots it has — lengthen the step before adding knots.
- Input bounds are where MPC earns its keep over LQR; set them from the actual actuator, not as slack.
- Warm-starting is not optional at these rates: from cold, the solver spends its iteration budget re-finding the previous answer.

## Failure Modes and Diagnostics

- A constant lag proportional to reference speed means the preview is not wired up.
- Too few solver iterations shows up as jitter, not as an error message.
- Bounds tight enough to make the problem infeasible produce whatever the solver returns on failure.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.mpc_tracking
```

## Evidence

![MPC Tracking](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/mpc_tracking/mpc_tracking.gif)

## References

- [Rawlings, Mayne, Diehl, Model Predictive Control: Theory, Computation, and Design](https://sites.engineering.ucsb.edu/~jbraw/mpc/)
- [Borrelli, Bemporad, Morari, Predictive Control for Linear and Hybrid Systems](https://www.cambridge.org/core/books/predictive-control-for-linear-and-hybrid-systems/1E6E2C5B2C1C1BB2E1F0E9EE6C0E6B7A)

## Related Algorithms

- [LQR Path Tracking](/simulations/path-tracking/lqr-tracking)
- [NMPC](/simulations/trajectory-tracking/nmpc)
- [MPPI](/simulations/trajectory-tracking/mppi)
