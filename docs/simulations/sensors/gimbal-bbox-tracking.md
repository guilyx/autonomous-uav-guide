<!-- Erwin Lejeune — 2026-02-23 -->
# Gimbal Bounding Box Tracking

## Algorithm

PD controller with EMA filtering that centres a bounding box in the camera frame. Handles target loss and re-acquisition.

**Reference:** F. Chaumette & S. Hutchinson, "Visual Servo Control — Part II: Advanced Approaches," IEEE RAM, 2007.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| kp | 1.2 | Proportional gain |
| kd | 0.5 | Derivative damping |
| EMA α | 0.15 | Smoothing coefficient |

## Run

```bash
python -m uav_sim.simulations.sensors.gimbal_bbox_tracking
```
