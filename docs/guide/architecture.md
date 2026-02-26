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

## Simulation Composition API (Quad-First)

`uav_sim.simulations` now exposes a composition-first API so users can assemble scenarios without editing internals:

1. Create environment
2. Spawn platform
3. Define mission
4. Run simulation

```python
from pathlib import Path

from uav_sim.simulations import (
    create_environment,
    create_mission,
    create_sim,
    run_sim,
    spawn_quad_platform,
)

env = create_environment(n_buildings=0, world_size=30.0)
platform = spawn_quad_platform()
mission = create_mission(path=my_path, standard=my_standard, fallback_policy="none")
sim = create_sim(name="my_demo", out_dir=Path("."), mission=mission)
result = run_sim(sim=sim, env=env, platform=platform)
```

### Contracts for extensibility

- `PayloadPlugin`: attach sensors/payload modules that emit standardized packets.
- `PathPlannerPlugin`, `TrackerPlugin`, `EstimatorPlugin`, `PerceptionPlugin`: pluggable algorithm interfaces.
- `QuadPlatform`: current reference platform implementation; fixed-wing and VTOL can be added via adapters without changing mission APIs.
