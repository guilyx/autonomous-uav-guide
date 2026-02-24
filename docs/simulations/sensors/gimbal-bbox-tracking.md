<!-- Erwin Lejeune — 2026-02-24 -->
# Gimbal Bounding Box Tracking

## Problem Statement

Bounding-box tracking aligns camera orientation to maintain target centering and approximate scale regulation in image space.
It is a practical visual-tracking controller when depth is partially observable.

## Model and Formulation

Let `(u,v)` be target center and `(u^*,v^*)` desired image center.
Error:

$$
e = [u-u^*, v-v^*]^\top
$$

PD control with smoothing:

$$
u_{cmd}=K_p e + K_d \dot{e},\quad e_f = \alpha e + (1-\alpha)e_f^{-}
$$

## Algorithm Procedure

1. Detect target bounding box each frame.
2. Compute center error and filtered derivatives.
3. Convert error to pan-tilt command increments.
4. Handle target loss with hold-and-search behavior.

## Tuning and Failure Modes

- Large derivative gain amplifies detector jitter.
- Heavy filtering reduces noise but adds tracking lag.
- Persistent target dropout requires robust reacquisition logic.

## Implementation and Execution

```bash
python -m uav_sim.simulations.sensors.gimbal_bbox_tracking
```

## Evidence

![Gimbal BBox Tracking](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/sensors/gimbal_bbox_tracking/gimbal_bbox_tracking.gif)

## References

- [Chaumette and Hutchinson, Visual Servo Control Part II (2007)](https://doi.org/10.1109/MRA.2007.339609)
- [Szeliski, Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)

## Related Algorithms

- [Gimbal Tracking](/simulations/sensors/gimbal-tracking)
- [Visual Servoing](/simulations/perception/visual-servoing)
