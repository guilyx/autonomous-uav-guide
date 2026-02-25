<!-- Erwin Lejeune — 2026-02-24 -->
# Quintic Polynomial Trajectory

## Problem Statement

Quintic trajectories are a standard choice for smooth motion generation with guaranteed continuity up to acceleration.
They provide practical C2 motion profiles for UAV transitions and corridor following.

## Model and Formulation

Use a fifth-order polynomial:

$$
p(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5
$$

Boundary conditions at initial and final times constrain position, velocity, and acceleration:

$$
p(0),\dot{p}(0),\ddot{p}(0),p(T),\dot{p}(T),\ddot{p}(T)
$$

forming a linear system for `a_0..a_5`.

## Algorithm Procedure

1. Select endpoint state and maneuver duration `T`.
2. Construct 6x6 boundary-condition system.
3. Solve coefficients and evaluate trajectory over time.
4. Feed resulting reference to the tracking controller.

## Tuning Guidance

- Duration `T` controls aggressiveness directly.
- Use axis-specific duration scaling for asymmetric constraints.
- Validate peak acceleration and jerk against actuator limits.

## Failure Modes and Diagnostics

- Very small `T` values produce infeasible accelerations.
- Discontinuous waypoint chaining can violate C2 assumptions.
- Numerical conditioning worsens when time scales vary widely.

## Implementation and Execution

```bash
python -m uav_sim.simulations.trajectory_planning.quintic_polynomial_demo
```

## Evidence

![Quintic Polynomial](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/quintic_polynomial_demo/quintic_polynomial_demo.gif)

## References

- [Werling et al., Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame (2010)](https://doi.org/10.1109/ROBOT.2010.5509799)
- [Kelly and Nagy, Reactive Nonholonomic Trajectory Generation via Parametric Optimal Control](https://doi.org/10.1177/0278364013491292)

## Related Algorithms

- [Polynomial Trajectory](/simulations/trajectory-planning/polynomial)
- [Frenet Optimal](/simulations/trajectory-planning/frenet-optimal)
