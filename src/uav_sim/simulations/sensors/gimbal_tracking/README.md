# Gimbal FOV Tracking

A two-axis gimbal model with rate-limited servos tracks a ground target from a hovering drone. The camera FOV frustum is visualised as the gimbal sweeps across a coverage path, demonstrating the look-at kinematics.

## Key Equations

$$\text{pitch}_{\text{des}} = \arctan\frac{-(t_z - p_z)}{\|t_{xy} - p_{xy}\|}, \quad \text{yaw}_{\text{des}} = \text{atan2}(t_y - p_y, t_x - p_x)$$

## Reference

D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA 2011. [DOI](https://doi.org/10.1109/ICRA.2011.5980409)

## Usage

```bash
python -m uav_sim.simulations.sensors.gimbal_tracking
```

## Result

![gimbal_tracking](gimbal_tracking.gif)
