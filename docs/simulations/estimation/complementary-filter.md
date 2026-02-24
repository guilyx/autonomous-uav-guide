<!-- Erwin Lejeune — 2026-02-23 -->
# Complementary Filter

## Algorithm

Fuses accelerometer (low-frequency, drift-free) and gyroscope (high-frequency, smooth) data using a weighted combination in the frequency domain.

**Reference:** R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear Complementary Filters on the Special Orthogonal Group," IEEE TAC, 2008.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Alpha | 0.98 | Gyroscope trust weight |
| IMU rate | 200 Hz | Sensor update rate |
| GPS noise | 0.5 m | Position measurement |

## Run

```bash
python -m uav_sim.simulations.estimation.complementary_filter
```
