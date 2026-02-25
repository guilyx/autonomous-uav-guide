<!-- Erwin Lejeune — 2026-02-24 -->
# Visual Servoing

## Problem Statement

Visual servoing controls UAV motion directly from image-space error signals.
It is effective for target following and precision alignment when full 3D reconstruction is unavailable.

## Model and Formulation

Given image feature error `e = s - s^*`, the control law is:

$$
\dot{q} = -\lambda L_s^+ e
$$

where `L_s` is the interaction matrix and `L_s^+` its pseudo-inverse.
In bounding-box tracking, feature vectors include center and area terms.

## Algorithm Procedure

1. Extract target feature in image frame.
2. Compute feature error to desired setpoint.
3. Convert image-space error to body-frame commands.
4. Apply velocity/attitude commands with saturation limits.

## Tuning and Failure Modes

- Gain `\lambda` too high causes oscillatory camera motion.
- Target occlusion can destabilize command generation without fallback logic.
- Camera latency and rolling shutter distort high-speed tracking.

## Implementation and Execution

```bash
python -m uav_sim.simulations.perception.visual_servoing
```

## Evidence

![Visual Servoing](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/visual_servoing/visual_servoing.gif)

## References

- [Chaumette and Hutchinson, Visual Servo Control Part I (2006)](https://doi.org/10.1109/MRA.2006.250573)
- [Chaumette and Hutchinson, Visual Servo Control Part II (2007)](https://doi.org/10.1109/MRA.2007.339609)

## Related Algorithms

- [Gimbal Tracking](/simulations/sensors/gimbal-tracking)
- [Gimbal BBox Tracking](/simulations/sensors/gimbal-bbox-tracking)
