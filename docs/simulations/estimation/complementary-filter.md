<!-- Erwin Lejeune — 2026-02-24 -->
# Complementary Filter

## Problem Statement

Low-cost IMU attitude estimation suffers from gyroscope drift and accelerometer noise.
The complementary filter combines high-frequency gyroscope integration with low-frequency gravity alignment to produce stable orientation estimates for downstream control.

## Model and Formulation

Let `\omega_m` be measured angular rate and `a_m` the measured acceleration.
The roll/pitch estimate is propagated by integrating rates, then corrected with an accelerometer-derived attitude:

$$
\theta_k = \alpha \left(\theta_{k-1} + \omega_{m,k}\Delta t\right) + (1-\alpha)\theta_{acc,k}
$$

where `\alpha \in (0, 1)` defines the crossover between gyro and accelerometer trust.

## Algorithm Procedure

1. Integrate gyroscope rates to obtain predicted orientation.
2. Compute gravity-consistent roll and pitch from accelerometer measurements.
3. Blend the two estimates using the complementary gain `\alpha`.
4. Output attitude estimate to attitude and position controllers.

## Tuning Guidance

- `\alpha` near `0.98` works well when IMU rates are high (`>= 100 Hz`).
- Increase accelerometer weighting when gyro bias grows over long flights.
- Reduce accelerometer weighting in aggressive maneuvers where linear acceleration contaminates gravity direction.

## Failure Modes and Diagnostics

- Sustained acceleration can be misinterpreted as tilt, causing attitude bias.
- Poor gyro bias handling leads to slow heading/attitude drift.
- Check innovation between gyro-integrated and accelerometer attitude; persistent bias indicates gain mismatch.

## Implementation and Execution

```bash
python -m uav_sim.simulations.estimation.complementary_filter
```

## Evidence

![Complementary Filter](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/complementary_filter/complementary_filter.gif)

## References

- [Mahony, Hamel, Pflimlin, Nonlinear Complementary Filters on SO(3), IEEE TAC (2008)](https://doi.org/10.1109/TAC.2008.923738)
- [Sabatini, Quaternion-based complementary filter review (2011)](https://doi.org/10.3390/s110201482)

## Related Algorithms

- [Extended Kalman Filter](/simulations/estimation/ekf)
- [GPS-IMU Fusion](/simulations/estimation/gps-imu-fusion)
