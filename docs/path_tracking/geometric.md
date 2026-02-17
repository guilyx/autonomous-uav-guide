# Geometric Controller on SO(3)

## Theory

Operates directly on the rotation group SO(3), avoiding singularities inherent in Euler-angle representations. The attitude error is defined via the vee map:

$$e_R = \frac{1}{2}(R_d^\top R - R^\top R_d)^\vee$$

Thrust is computed from position error; body torques from attitude and angular rate errors with PD gains.

## Reference

T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," CDC, 2010. [DOI: 10.1109/CDC.2010.5717652](https://doi.org/10.1109/CDC.2010.5717652)

## Simulation

```bash
uv run python simulations/path_tracking/geometric_control/run.py
```

![Geometric Control](../../simulations/path_tracking/geometric_control/geometric_control.gif)

## API

```python
from uav_sim.path_tracking.geometric_controller import GeometricController
ctrl = GeometricController()
wrench = ctrl.compute(state_12, target_position_3)
```
