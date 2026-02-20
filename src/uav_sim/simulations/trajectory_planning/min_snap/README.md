# Minimum-Snap Trajectory

Generates smooth trajectories by minimising the integral of the squared snap (4th derivative of position) through a sequence of waypoints. Produces dynamically feasible, C4-continuous trajectories ideal for quadrotors.

## Key Equations

$$\min \int_0^T \left\|\frac{d^4 p}{dt^4}\right\|^2 dt \quad \text{s.t. } p(t_i) = w_i$$

## Reference

D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA 2011. [DOI](https://doi.org/10.1109/ICRA.2011.5980409)

## Usage

```bash
python -m uav_sim.simulations.trajectory_planning.min_snap
```

## Result

![min_snap](min_snap.gif)
