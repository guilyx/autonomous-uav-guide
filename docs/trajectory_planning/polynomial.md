# Polynomial Trajectory

## Theory

Fits piecewise quintic (or higher-order) polynomials through waypoints with continuity constraints on position, velocity, and acceleration. Each segment is parameterised as:

$$\mathbf{p}(t) = \sum_{k=0}^{n} \mathbf{c}_k t^k$$

The coefficients are found by solving a linear system enforcing boundary and interior waypoint constraints.

## Reference

C. Richter, A. Bry, N. Roy, "Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Indoor Environments," ISRR, 2013. [DOI: 10.1007/978-3-319-28872-7_37](https://doi.org/10.1007/978-3-319-28872-7_37)

## Simulation

```bash
uv run python simulations/trajectory_planning/polynomial_trajectory/run.py
```

![Polynomial Trajectory](../../simulations/trajectory_planning/polynomial_trajectory/polynomial_trajectory.gif)

## API

```python
from uav_sim.path_planning.polynomial_trajectory import PolynomialTrajectory
poly = PolynomialTrajectory(order=5)
coeffs = poly.generate(waypoints, segment_times)
times, positions = poly.evaluate(coeffs, segment_times, dt=0.01)
```
