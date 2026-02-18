# Waypoint Tracking

Sequential waypoint tracking with a cascaded PID controller. The drone navigates through an ordered list of 3D waypoints, transitioning to the next when within a capture radius.

## Key Equations

$$\text{if } \|p - w_i\| < r_{\text{capture}} \text{ then } i \leftarrow i + 1$$

## Reference

R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and Practice," Princeton University Press, 2012.

## Usage

```bash
python -m uav_sim.simulations.path_tracking.waypoint_tracking
```

## Result

![waypoint_tracking](waypoint_tracking.gif)
