<!-- Erwin Lejeune — 2026-02-24 -->
# Unscented Kalman Filter (UKF)

## Problem Statement

For strongly nonlinear dynamics, first-order Jacobian linearization can degrade EKF accuracy.
The UKF uses deterministic sigma-point propagation to capture higher-order effects without symbolic Jacobians.

## Model and Formulation

Given state mean `\mu` and covariance `P`, construct `2n+1` sigma points:

$$
\chi_0 = \mu,\quad \chi_i = \mu \pm \left(\sqrt{(n+\lambda)P}\right)_i
$$

Propagate each point through nonlinear models, then recover moments:

$$
\hat{\mu} = \sum_i W_i^{(m)}\chi_i,\quad \hat{P} = \sum_i W_i^{(c)}(\chi_i-\hat{\mu})(\chi_i-\hat{\mu})^\top + Q
$$

## Algorithm Procedure

1. Generate sigma points using `(\alpha,\beta,\kappa)` scaling.
2. Propagate points through process model for prediction.
3. Project predicted points into measurement space.
4. Compute gain from cross-covariance and update posterior state.

## What This Simulation Measures

GPS is unavailable; the drone localises from noisy **ranges to four
surveyed ground anchors**. That measurement model is nonlinear in the
state, which is what makes it a UKF demo — given a linear model the UKF
reduces exactly to the Kalman filter and there is nothing to show.

## Tuning Guidance

- Use small `\alpha` (`1e-3` to `1e-1`) for local spread control.
- Set `\beta=2` for approximately Gaussian priors.
- Increase process noise if sigma clouds collapse under model mismatch.
- **Scale `Q` with the step size.** `Q` is the covariance accumulated over
  one step. A fixed `diag([...])` at 200 Hz claims the velocity
  random-walks by 0.32 m/s every 5 ms; the filter concludes its own
  prediction is worthless and degenerates into echoing the measurement.
  Build it from an acceleration noise density instead
  (`uav_sim.estimation.constant_velocity_q`), and the tuning survives a
  change of rate.
- Sanity-check the reported 1σ against the actual error. A filter whose
  covariance is much tighter than its error has stopped listening.

## Failure Modes and Diagnostics

- Poor scaling parameters can produce non-positive definite covariance.
- Non-Gaussian heavy-tailed noise can still break Gaussian-moment assumptions.
- Monitor covariance eigenvalues to detect numerical instability.

## Implementation and Execution

```bash
python -m uav_sim.simulations.estimation.ukf
```

## Evidence

![UKF](https://media.githubusercontent.com/media/guilyx/flybots/main/src/uav_sim/simulations/estimation/ukf/ukf.gif)

## References

- [Wan and Van der Merwe, The Unscented Kalman Filter for Nonlinear Estimation (2000)](https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf)
- [Julier and Uhlmann, Unscented Filtering and Nonlinear Estimation, Proceedings of the IEEE (2004)](https://doi.org/10.1109/JPROC.2003.823141)

## Related Algorithms

- [Extended Kalman Filter](/simulations/estimation/ekf)
- [Particle Filter](/simulations/estimation/particle-filter)
