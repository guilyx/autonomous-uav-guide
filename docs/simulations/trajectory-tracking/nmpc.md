<!-- Erwin Lejeune — 2026-02-24 -->
# Nonlinear Model Predictive Control (NMPC)

## Problem Statement

NMPC optimizes constrained control over a finite horizon using the full nonlinear UAV model.
It handles state and input constraints explicitly while tracking aggressive trajectories.

## Model and Formulation

At each control step solve:

$$
\min_{u_{0:N-1}} \sum_{k=0}^{N-1}\left(\|x_k-x_k^{ref}\|_Q^2 + \|u_k-u_k^{ref}\|_R^2\right) + \|x_N-x_N^{ref}\|_P^2
$$

subject to:

$$
x_{k+1}=f(x_k,u_k),\quad u_{min}\le u_k \le u_{max}
$$

## Algorithm Procedure

1. Warm-start control sequence from previous solution.
2. Integrate dynamics (single-shooting/multiple-shooting).
3. Solve nonlinear program under constraints.
4. Apply first control action, then shift horizon.

## What This Implementation Optimises Over

Decision variables are **collective thrust and body rates**, not torques —
the formulation used by most working quadrotor NMPC (Falanga et al., PAMPC).
The reason is a timescale conflict: torque moves attitude in tens of
milliseconds and position in seconds, so a horizon short enough to
integrate torque stably cannot see position at all, and one long enough to
see position integrates the rotational dynamics to infinity. Taking body
rates as inputs removes the stiff dynamics from the prediction model and
delegates them to a rate loop underneath, which is how a real flight stack
is layered. The prediction model is still fully nonlinear:

$$
\dot p = v,\qquad
\dot v = \frac{T}{m}R(\eta)e_3 - g e_3,\qquad
\dot\eta = W(\eta)\,\omega
$$

The prediction step is decoupled from the control step — predict coarsely
over ~1 s, re-plan at 20 Hz — and **move blocking** holds the input
constant over blocks of prediction steps. `L-BFGS-B` builds its gradient by
finite differences, so cost scales with the number of decision variables;
blocking buys horizon length nearly for free.

The reference is sampled forward over the horizon. Holding it at its
present value asks the plan to come to rest where the trajectory used to
be, which costs roughly half a horizon of lag no matter how well the
solver converges.

## Tuning Guidance

- Increase terminal weight `P` to improve horizon-end stability.
- Use shorter horizons for strict realtime budgets.
- Start with soft constraints before switching to hard constraints.
- Lengthen the *prediction* step before shortening the horizon: what
  matters is how far ahead the horizon reaches, not how many knots it has.

## Failure Modes and Diagnostics

- Solver infeasibility appears under inconsistent references or tight bounds.
- Poor warm starts increase optimization latency.
- Inaccurate models produce biased constraint activity.

## Implementation and Execution

```bash
python -m uav_sim.simulations.trajectory_tracking.nmpc
```

## Evidence

![NMPC](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/nmpc/nmpc.gif)

## References

- [Diehl et al., Real-Time Optimization and NMPC (2002)](https://doi.org/10.1016/S0959-1524(02)00058-9)
- [Rawlings, Mayne, Diehl, Model Predictive Control: Theory, Computation, and Design](https://sites.engineering.ucsb.edu/~jbraw/mpc/)

## Related Algorithms

- [MPPI](/simulations/trajectory-tracking/mppi)
- [LQR Hover](/simulations/path-tracking/lqr-hover)
