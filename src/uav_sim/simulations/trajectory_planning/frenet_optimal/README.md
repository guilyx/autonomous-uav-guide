# Frenet-Frame Optimal Trajectory

Generates candidate trajectories in the Frenet (road-aligned) frame by sampling lateral offsets and longitudinal velocities, then selects the minimum-cost feasible trajectory that avoids obstacles.

## Key Equations

$$\min_{\delta, T} J = k_j J_{\text{jerk}} + k_t T + k_d \delta_f^2 + k_s (s_f - s_{\text{target}})^2$$

## Reference

M. Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame," ICRA 2010. [DOI](https://doi.org/10.1109/ROBOT.2010.5509799)

## Usage

```bash
python -m uav_sim.simulations.trajectory_planning.frenet_optimal
```

## Result

![frenet_optimal](frenet_optimal.gif)
