<!-- Erwin Lejeune — 2026-02-24 -->
# GPS-IMU Fusion

## Problem Statement

IMU integration is high-rate but drifts; GPS is globally referenced but low-rate and noisy.
GPS-IMU fusion combines both into a single estimate that remains smooth, drift-limited, and real-time.

## Model and Formulation

The fused estimator uses IMU-driven prediction and GPS correction:

$$
\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}, u^{imu}_k)
$$

$$
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k\left(z^{gps}_k - h(\hat{x}_{k|k-1})\right)
$$

The asynchronous rate structure is critical: `f` runs at IMU rate, update only when GPS messages arrive.

## Algorithm Procedure

1. Integrate IMU acceleration and angular rate at high frequency.
2. Propagate covariance with process model and bias noise.
3. On each GPS arrival, compute innovation and perform correction.
4. Publish fused state for path tracking and planning.

## Tuning Guidance

- Keep GPS `R` realistic; over-trusting GPS causes noisy position estimates.
- Include IMU bias states whenever drift is non-negligible.
- Use timestamp-consistent interpolation/extrapolation for sensor alignment.

## Failure Modes and Diagnostics

- Time synchronization error can appear as oscillatory position correction.
- GPS dropouts cause rapid uncertainty growth if process noise is under-modeled.
- Bias unobservability yields slowly diverging velocity/attitude estimates.

## Implementation and Execution

```bash
python -m uav_sim.simulations.estimation.gps_imu_fusion
```

## Evidence

![GPS IMU Fusion](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/gps_imu_fusion/gps_imu_fusion.gif)

## References

- [Groves, Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems](https://www.artechhouse.com/Principles-of-GNSS-Inertial-and-Multisensor-Integrated-Navigation-Systems-Second-Edition-P1609.aspx)
- [Maybeck, Stochastic Models, Estimation, and Control](https://books.google.com/books?id=8GQ7DwAAQBAJ)

## Related Algorithms

- [Extended Kalman Filter](/simulations/estimation/ekf)
- [Complementary Filter](/simulations/estimation/complementary-filter)
