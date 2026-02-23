<!-- Erwin Lejeune — 2026-02-23 -->
# Architecture

## Quadrotor Model

The core physics engine is a 6DOF rigid-body quadrotor using Newton-Euler equations with RK4 integration.

**State vector** (12 dimensions):
- Position: `[x, y, z]`
- Orientation (Euler): `[φ, θ, ψ]`
- Linear velocity: `[u, v, w]`
- Angular velocity: `[p, q, r]`

**Control input** (4 dimensions):
- Total thrust `T`
- Body torques `[τ_x, τ_y, τ_z]`

## Control Stack

The cascaded control architecture follows PX4-inspired design:

```
Position Controller → Velocity Controller → Attitude Controller → Rate Controller → Motor Mixer
```

The `StateManager` implements a finite state machine:
`DISARMED → ARM → TAKEOFF → HOVER → {TRACKING, OFFBOARD, LOITER} → LAND → DISARMED`

## Sensors

All sensors simulate real-world noise characteristics:

| Sensor | Noise Model | Typical Rate |
|--------|-------------|-------------|
| GPS | Gaussian, 0.5m std | 10 Hz |
| IMU | Gyro bias + white noise | 200 Hz |
| Lidar2D/3D | Range noise + dropout | 10-40 Hz |
| Camera | Projection + distortion | 30 Hz |

## Estimation Pipeline

Filters fuse noisy sensor data into state estimates:

- **Complementary Filter**: Fast attitude estimation (accel + gyro)
- **EKF/UKF**: Full 6DOF state estimation with GPS/IMU fusion
- **Particle Filter**: Non-Gaussian posterior, GPS localisation
