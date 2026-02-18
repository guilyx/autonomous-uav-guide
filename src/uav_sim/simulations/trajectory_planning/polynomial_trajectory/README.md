# Polynomial Trajectory

Generates piecewise polynomial trajectories through waypoints with specified boundary conditions. Each segment is a 5th/7th order polynomial that satisfies position, velocity, and acceleration constraints at the knot points.

## Key Equations

$$p(t) = \sum_{k=0}^{n} a_k t^k, \quad p(t_i) = w_i, \; \dot p(t_i), \; \ddot p(t_i) \text{ continuous}$$

## Reference

C. Richter, A. Bry, N. Roy, "Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Indoor Environments," ISRR 2013. [DOI](https://doi.org/10.1007/978-3-319-28872-7_37)

## Usage

```bash
python -m uav_sim.simulations.trajectory_planning.polynomial_trajectory
```

## Result

![polynomial_trajectory](polynomial_trajectory.gif)
