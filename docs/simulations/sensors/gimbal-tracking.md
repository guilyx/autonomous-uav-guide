<!-- Erwin Lejeune — 2026-02-24 -->
# Gimbal Tracking

## Problem Statement

Gimbal pointing control keeps a camera line-of-sight centered on a moving target while the UAV body rotates and translates.

## Model and Formulation

Pan-tilt angle errors:

$$
e_{\psi} = \psi_{des}-\psi,\quad e_{\theta} = \theta_{des}-\theta
$$

Basic control law:

$$
u = K_p e
$$

Desired angles are generated from look-at geometry between camera and target coordinates.

## Algorithm Procedure

1. Compute target direction in camera/body frames.
2. Derive desired pan and tilt angles.
3. Apply proportional gimbal control with rate limits.
4. Feed stabilized image stream to vision modules.

## Tuning and Failure Modes

- Low gains under-track fast targets.
- High gains induce jitter near setpoint.
- Inaccurate extrinsics lead to persistent pointing offsets.

## Implementation and Execution

```bash
python -m uav_sim.simulations.sensors.gimbal_tracking
```

## Evidence

![Gimbal Tracking](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_tracking/gimbal_tracking.gif)

## References

- [Shin and Park, PID-based Gimbal Stabilization for UAV Camera Systems](https://doi.org/10.1109/ICCAS.2017.8204502)
- [Siciliano et al., Robotics: Modelling, Planning and Control](https://link.springer.com/book/10.1007/978-1-84628-642-1)

## Related Algorithms

- [Visual Servoing](/simulations/perception/visual-servoing)
- [Gimbal BBox Tracking](/simulations/sensors/gimbal-bbox-tracking)
