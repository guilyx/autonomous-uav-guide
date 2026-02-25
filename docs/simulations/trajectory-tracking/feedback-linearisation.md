<!-- Erwin Lejeune — 2026-02-24 -->
# Feedback Linearisation

## Problem Statement

Nonlinear quadrotor dynamics complicate direct trajectory tracking when translational and rotational channels are tightly coupled.
Feedback linearisation cancels known nonlinear terms and imposes target linear error dynamics.

## Model and Formulation

Given nonlinear dynamics `\dot{x}=f(x)+g(x)u`, define an output `y=h(x)` and choose:

$$
u = g(x)^{-1}\left(v - f(x)\right)
$$

to recover approximately linear closed-loop output dynamics:

$$
y^{(r)} = v
$$

Then choose `v` as a PD/PID law over trajectory tracking error.

## Algorithm Procedure

1. Compute desired derivatives from trajectory generator.
2. Evaluate model terms for drift and input coupling.
3. Cancel nonlinear dynamics via inverse model.
4. Apply linear error feedback in transformed coordinates.

## Tuning Guidance

- Use conservative gains under model uncertainty.
- Regularize inverse terms near singular operating conditions.
- Blend with robust feedback when aerodynamic disturbances dominate.

## Failure Modes and Diagnostics

- Model mismatch directly leaks into control error.
- Input inversion can amplify sensor noise.
- Singular regions in attitude parameterization can cause spikes.

## Implementation and Execution

```bash
python -m uav_sim.simulations.trajectory_tracking.feedback_linearisation
```

## Evidence

![Feedback Linearisation](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_tracking/feedback_linearisation/feedback_linearisation.gif)

## References

- [Mellinger and Kumar, Minimum Snap Trajectory Generation and Control for Quadrotors (2011)](https://doi.org/10.1109/ICRA.2011.5980409)
- [Isidori, Nonlinear Control Systems](https://www.springer.com/gp/book/9781846286148)

## Related Algorithms

- [Minimum Snap](/simulations/trajectory-planning/min-snap)
- [NMPC](/simulations/trajectory-tracking/nmpc)
