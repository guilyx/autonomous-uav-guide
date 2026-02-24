<!-- Erwin Lejeune — 2026-02-23 -->
# Extended Kalman Filter (EKF)

## Algorithm

The EKF linearises the nonlinear process and measurement models around the current estimate using first-order Taylor expansion (Jacobians). It runs a predict-update cycle at each timestep.

**Reference:** S. Julier, J. Uhlmann, "A New Extension of the Kalman Filter to Nonlinear Systems," AeroSense, 1997.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| State dim | 4 | `[x, y, vx, vy]` |
| GPS noise | 0.5 m | Measurement noise std |
| dt | 0.005 s | Simulation timestep |

## Run

```bash
python -m uav_sim.simulations.estimation.ekf
```
