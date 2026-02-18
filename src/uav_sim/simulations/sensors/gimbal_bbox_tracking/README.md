# Gimbal Bounding Box Tracking

A stationary hovering drone tracks a moving ground target using only
gimbal pan/tilt control. The BBoxTracker controller keeps the target's
bounding box centred in the camera image frame.

## Key Equations

Proportional gimbal rate commands:

$$\dot{\theta}_{\text{pan}} = K_p \cdot e_x, \quad \dot{\theta}_{\text{tilt}} = K_p \cdot e_y$$

where $(e_x, e_y)$ are the normalised image-space errors of the bounding
box centre.

## Reference

F. Chaumette & S. Hutchinson, "Visual Servo Control — Part II: Advanced
Approaches," IEEE Robotics & Automation Magazine, vol. 14, no. 1, 2007.
[DOI](https://doi.org/10.1109/MRA.2007.339609)

## Usage

```bash
python -m uav_sim.simulations.sensors.gimbal_bbox_tracking
```

## Result

![gimbal_bbox_tracking](gimbal_bbox_tracking.gif)
