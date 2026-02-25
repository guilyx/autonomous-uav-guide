<!-- Erwin Lejeune — 2026-02-24 -->
# Extended Kalman Filter (EKF)

## Problem Statement

The EKF estimates nonlinear UAV state with Gaussian uncertainty when direct closed-form Bayesian updates are unavailable.
It is commonly used for tightly-coupled inertial and positional sensing in real-time flight stacks.

## Model and Formulation

Nonlinear system:

$$
x_k = f(x_{k-1}, u_k) + w_k,\quad z_k = h(x_k) + v_k
$$

Linearization around the current estimate yields Jacobians `F_k = \partial f/\partial x`, `H_k = \partial h/\partial x`.
The covariance recursion is:

$$
P_{k|k-1} = F_k P_{k-1|k-1} F_k^\top + Q_k
$$

$$
K_k = P_{k|k-1}H_k^\top(H_kP_{k|k-1}H_k^\top + R_k)^{-1}
$$

## Algorithm Procedure

1. Predict state with nonlinear process model `f`.
2. Propagate covariance using local Jacobian `F_k`.
3. Compute innovation `y_k = z_k - h(\hat{x}_{k|k-1})`.
4. Update state and covariance with Kalman gain `K_k`.

## Tuning Guidance

- Start with conservative `Q` to avoid overconfident predictions.
- Increase `R` for noisy GPS updates in urban or multipath environments.
- Validate filter consistency using normalized innovation squared (NIS).

## Failure Modes and Diagnostics

- Linearization error can destabilize updates during aggressive maneuvers.
- Unmodeled bias states produce persistent innovation drift.
- Divergence often appears as shrinking covariance but rising position error.

## Implementation and Execution

```bash
python -m uav_sim.simulations.estimation.ekf
```

## Evidence

![EKF](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/ekf/ekf.gif)

## References

- [Julier and Uhlmann, New Extension of the Kalman Filter to Nonlinear Systems (1997)](https://ieeexplore.ieee.org/document/799967)
- [Maybeck, Stochastic Models, Estimation, and Control, Volume 1](https://books.google.com/books?id=8GQ7DwAAQBAJ)

## Related Algorithms

- [Unscented Kalman Filter](/simulations/estimation/ukf)
- [GPS-IMU Fusion](/simulations/estimation/gps-imu-fusion)
