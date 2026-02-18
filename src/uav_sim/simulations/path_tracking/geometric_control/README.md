# Geometric Control on SO(3)

Tracks position/attitude using a controller formulated directly on the rotation group SO(3), avoiding Euler angle singularities. The attitude error is computed from the rotation matrix difference, with feed-forward angular velocity compensation.

## Key Equations

$$e_R = \\frac{1}{2}(R_d^T R - R^T R_d)^\\vee, \\quad \\tau = -k_R e_R - k_\\omega e_\\omega + \\omega \\times J\\omega$$

## Reference

T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," CDC 2010. [DOI](https://doi.org/10.1109/CDC.2010.5717652)

## Usage

```bash
python -m uav_sim.simulations.path_tracking.geometric_control
```

## Result

![geometric_control](geometric_control.gif)
