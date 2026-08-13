<!-- Erwin Lejeune — 2026-08-13 -->
# Geometric Control on SE(3)

## Problem Statement

Euler-angle controllers carry singularities and lose meaning at large attitudes.
Geometric control works directly on the rotation group `SO(3)`, so the attitude error is well defined everywhere and the same law covers gentle hover and aggressive flight.

## Model and Formulation

Position control produces a desired force vector:

$$
F_d = m\left(-k_x e_p - k_v e_v + g e_3 + \ddot{p}_{ref}\right)
$$

Thrust is its projection onto the current body z-axis, and the desired attitude is built from `F_d`:

$$
T = F_d \cdot R e_3, \qquad b_{3d} = \frac{F_d}{\lVert F_d \rVert}
$$

The attitude error lives in the Lie algebra:

$$
e_R = \tfrac{1}{2}\left(R_d^\top R - R^\top R_d\right)^{\vee},
\qquad
e_\Omega = \Omega - R^\top R_d\,\Omega_d
$$

$$
M = -k_R e_R - k_\Omega e_\Omega + \Omega \times J\Omega - J\left(\hat{\Omega}R^\top R_d \Omega_d\right)
$$

## Algorithm Procedure

1. Form position and velocity errors against the reference.
2. Build the desired force, adding the reference acceleration as feed-forward.
3. Extract thrust and the desired rotation `R_d` from that force.
4. Differentiate `R_d` to recover the desired angular velocity `Ω_d`.
5. Compute the `SO(3)` attitude error and the resulting torque.

## Two Terms That Are Easy to Drop

Both feed-forward terms are optional in the sense that the code runs without them, and both change the character of the controller when they are missing.

**Reference acceleration.** Without it the error dynamics are driven by
`p̈_ref` itself, so the tracking error is whatever the second-order error
system does when forced at the trajectory's own frequency. On the atlas
figure-8 that was a **17 % overshoot on the fast axis** and 0.39 m mean
error; adding it takes the same controller to 0.023 m.

**Desired angular velocity.** `Ω_d` is not zero on a moving trajectory —
`R_d` rotates as the required force vector swings around. Assuming
`Ω_d = 0` asks the attitude loop to fight its own reference. The
implementation recovers it by differentiating `R_d`:
`Ω̂_d ≈ R_d(k-1)^\top \dot{R}_d`.

## Tuning Guidance

Gains here are **derived from the inertia**, not typed in:

$$
k_R = J\omega_{att}^2, \qquad k_\Omega = 2\zeta J \omega_{att}
$$

- A gain that suits one airframe is wrong for another by the ratio of their inertias — the controller commands a torque, not an acceleration.
- Keep the attitude bandwidth several times the position bandwidth. If the two are comparable they resonate, and the quadrotor overshoots every turn of the reference. The default is 8×.
- `max_acc` bounds the horizontal acceleration, and therefore the commanded tilt.

## Failure Modes and Diagnostics

- Comparable position and attitude bandwidths produce a lightly damped overshoot that looks like a tuning problem but is a loop-separation problem.
- `F_d` near zero makes `b_3d` ill-conditioned; the implementation floors the norm.
- A desired yaw parallel to `b_3d` degenerates the `b_2d` cross product.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.geometric_control
```

## Evidence

Mean tracking error on the standard figure-8: **0.023 m**, peak 0.034 m, with the flown amplitude matching the reference to within 0.3 %.

![Geometric SO(3)](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/geometric_control/geometric_control.gif)

## References

- [Lee, Leok, McClamroch, Geometric Tracking Control of a Quadrotor UAV on SE(3) (2010)](https://doi.org/10.1109/CDC.2010.5717652)
- [Bullo and Lewis, Geometric Control of Mechanical Systems](https://link.springer.com/book/10.1007/978-1-4899-7276-7)

## Related Algorithms

- [Feedback Linearisation](/simulations/trajectory-tracking/feedback-linearisation)
- [LQR Path Tracking](/simulations/path-tracking/lqr-tracking)
- [NMPC](/simulations/trajectory-tracking/nmpc)
