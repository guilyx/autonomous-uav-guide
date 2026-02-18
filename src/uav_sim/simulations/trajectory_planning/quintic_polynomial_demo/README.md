# Quintic Polynomial Trajectory

Demonstrates a single-segment quintic polynomial that satisfies full boundary conditions (position, velocity, acceleration) at both endpoints. Produces the smoothest possible path for given start/end constraints.

## Key Equations

$$p(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

## Reference

K. G. Shin, N. D. McKay, "Minimum-Time Control of Robotic Manipulators with Geometric Path Constraints," IEEE TAC, 1985. [DOI](https://doi.org/10.1109/TAC.1985.1104009)

## Usage

```bash
python -m uav_sim.simulations.trajectory_planning.quintic_polynomial_demo
```

## Result

![quintic_polynomial_demo](quintic_polynomial_demo.gif)
