# Minimum-Snap Trajectory

## Theory

Generates smooth trajectories by minimising the integral of the squared snap (4th derivative of position), exploiting the *differential flatness* of quadrotor dynamics. The trajectory is represented as a piecewise 7th-order polynomial, and the optimisation is solved via a QP:

$$\min \int_0^T \|\mathbf{x}^{(4)}(t)\|^2 dt$$

subject to waypoint and continuity constraints.

## Reference

D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA, 2011. [DOI: 10.1109/ICRA.2011.5980409](https://doi.org/10.1109/ICRA.2011.5980409)

## Simulation

```bash
uv run python simulations/trajectory_planning/min_snap/run.py
```

![Min-Snap](../../simulations/trajectory_planning/min_snap/min_snap.gif)

## API

```python
from uav_sim.path_planning.min_snap import MinSnapTrajectory
ms = MinSnapTrajectory()
coeffs = ms.generate(waypoints, segment_times)
times, positions = ms.evaluate(coeffs, segment_times, dt=0.01)
```
