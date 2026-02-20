# LQR Path Tracking

Extends LQR hover to track a sequence of waypoints. At each step, the LQR regulator drives the quadrotor toward the nearest waypoint, advancing to the next when within a threshold distance.

## Key Equations

$$u_k = -K(x_k - x_{\text{wp},i}) + u_{\text{hover}}$$

## Reference

B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear Quadratic Methods," Prentice Hall, 1990.

## Usage

```bash
python -m uav_sim.simulations.path_tracking.lqr_tracking
```

## Result

![lqr_tracking](lqr_tracking.gif)
