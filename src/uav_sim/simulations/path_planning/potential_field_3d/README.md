# Artificial Potential Field 3D

Navigates by following the negative gradient of an artificial potential that combines an attractive goal potential with repulsive obstacle potentials. Simple and reactive but can get stuck in local minima.

## Key Equations

$$F = -\\nabla U, \\quad U = U_{\\text{att}} + U_{\\text{rep}} = \\frac{1}{2}k_a\\|x - x_g\\|^2 + \\sum_i \\frac{k_r}{2}\\left(\\frac{1}{d_i} - \\frac{1}{d_0}\\right)^2$$

## Reference

O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots," IJRR, 1986. [DOI](https://doi.org/10.1177/027836498600500106)

## Usage

```bash
python -m uav_sim.simulations.path_planning.potential_field_3d
```

## Result

![potential_field_3d](potential_field_3d.gif)
