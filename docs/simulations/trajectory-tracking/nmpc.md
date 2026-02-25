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

## Tuning Guidance

- Increase terminal weight `P` to improve horizon-end stability.
- Use shorter horizons for strict realtime budgets.
- Start with soft constraints before switching to hard constraints.

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
