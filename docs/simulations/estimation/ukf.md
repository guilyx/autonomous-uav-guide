<!-- Erwin Lejeune — 2026-02-23 -->
# Unscented Kalman Filter (UKF)

## Algorithm

Uses sigma points to propagate the mean and covariance through nonlinear models without computing Jacobians, providing better accuracy than EKF for highly nonlinear systems.

**Reference:** E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for Nonlinear Estimation," AS-SPCC, 2000.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 0.001 | Sigma point spread |
| β | 2.0 | Prior distribution (Gaussian) |
| κ | 0.0 | Secondary scaling |

## Run

```bash
python -m uav_sim.simulations.estimation.ukf
```
