# Artificial Potential Field

## Theory

Combines an attractive potential pulling the agent towards the goal with repulsive potentials pushing it away from obstacles:

$$U_{\text{att}} = \frac{1}{2}\zeta\|\mathbf{q} - \mathbf{q}_g\|^2$$

$$U_{\text{rep}} = \begin{cases}\frac{1}{2}\eta\left(\frac{1}{\rho} - \frac{1}{\rho_0}\right)^2 & \rho \le \rho_0\\0 & \rho > \rho_0\end{cases}$$

The robot follows the negative gradient of the total potential. Local minima are escaped via random perturbations.

## Reference

O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots," IJRR, 1986. [DOI: 10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106)

## Simulation

```bash
uv run python simulations/path_planning/potential_field_3d/run.py
```

![Potential Field 3D](../../simulations/path_planning/potential_field_3d/potential_field_3d.gif)

## API

```python
from uav_sim.path_planning.potential_field_3d import PotentialField3D
planner = PotentialField3D(zeta=1.0, eta=100.0, rho0=2.0)
path = planner.plan(start, goal, obstacles)
```
