<!-- Erwin Lejeune — 2026-02-23 -->
# Gimbal Point Tracking

## Algorithm

P-control loop that commands a pan-tilt gimbal to track a moving ground target along a coverage path. Uses `look_at` geometry for desired angles.

## Run

```bash
python -m uav_sim.simulations.sensors.gimbal_tracking
```
