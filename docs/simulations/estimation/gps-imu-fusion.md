<!-- Erwin Lejeune — 2026-02-23 -->
# GPS-IMU Fusion

## Algorithm

Combines GPS position updates with IMU dead-reckoning using an EKF. The IMU provides high-rate predictions while GPS corrects drift at lower rates.

**Reference:** P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems," Artech House, 2013.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| GPS rate | 10 Hz | Position update rate |
| IMU rate | 200 Hz | Prediction rate |
| GPS std | 0.5 m | Position noise |

## Run

```bash
python -m uav_sim.simulations.estimation.gps_imu_fusion
```
