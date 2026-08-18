<!-- Erwin Lejeune — 2026-08-13 -->
# LQR Path Tracking

## Problem Statement

[LQR hover](/simulations/path-tracking/lqr-hover) stabilises a fixed point.
Path tracking asks the same optimal feedback to follow a *moving* reference — which is not the problem LQR solves, and the gap between the two is where most of the tracking error comes from.

## Model and Formulation

The gain comes from the same hover linearisation and Riccati solution:

$$
u = u_{hover} - K\left(x - x_{ref}\right), \qquad K = R^{-1}B^\top P
$$

## An LQR Is a Regulator

This matters more than the weighting matrices. LQR drives the error to
zero only for references the model can hold with the equilibrium input.
A constant-velocity reference qualifies — `ẋ_ref = A x_ref` holds when
the velocity rows are already in `A`. An **accelerating** reference does
not: nothing in `u_hover` produces `p̈_ref`, so the loop has to generate
that acceleration out of error alone, and settles at whatever standing
error does the job.

The fix is the quadrotor's differential flatness written out. From the
hover linearisation, `v̇x = gθ` and `v̇y = -gφ`, so the attitude and
thrust that produce a commanded acceleration follow in closed form:

$$
\theta_{ff} = \frac{a_x}{g}, \qquad
\phi_{ff} = -\frac{a_y}{g}, \qquad
T_{ff} = m\,a_z
$$

Pass them through `feedforward_acc` and the regulator only has to correct
the residual:

```python
target = np.zeros(12)
target[:3], target[6:9] = ref_pos, ref_vel
wrench = lqr.compute(state, target, feedforward_acc=ref_acc)
```

On the atlas figure-8 that is the difference between **0.158 m** mean
error with a 4.6 % amplitude overshoot and **0.017 m** with none.

## Algorithm Procedure

1. Linearise about hover and solve the continuous-time Riccati equation once, offline.
2. Each step, build the reference state from the trajectory's position and velocity.
3. Convert the reference acceleration into feed-forward attitude and thrust.
4. Apply `u = u_hover + u_ff - K(x - x_ref)`.

## Tuning Guidance

- `Q` and `R` set the regulator's disturbance rejection, not its ability to follow a trajectory. If tracking error scales with reference speed, the missing piece is feed-forward, not gain.
- Re-linearise if the trajectory demands attitudes far from hover.
- The gain is computed once at construction; the Riccati solve is not in the loop.

## Failure Modes and Diagnostics

- Error proportional to reference velocity or acceleration means missing feed-forward.
- Aggressive trajectories violate the small-angle linearisation; the geometric controller has no such limit.
- Badly scaled `Q`/`R` units produce gains that look reasonable and behave strangely.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.lqr_tracking
```

## Evidence

![LQR Tracking](https://media.githubusercontent.com/media/guilyx/flybots/main/src/uav_sim/simulations/path_tracking/lqr_tracking/lqr_tracking.gif)

## References

- [Anderson and Moore, Optimal Control: Linear Quadratic Methods](https://books.google.com/books?id=iYMqAQAAMAAJ)
- [Mellinger and Kumar, Minimum Snap Trajectory Generation and Control for Quadrotors (2011)](https://doi.org/10.1109/ICRA.2011.5980409)

## Related Algorithms

- [LQR Hover](/simulations/path-tracking/lqr-hover)
- [MPC Tracking](/simulations/path-tracking/mpc-tracking)
- [Geometric Control](/simulations/path-tracking/geometric-control)
