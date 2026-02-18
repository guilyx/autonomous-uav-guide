# NMPC with PRM Planning

Combines Probabilistic Roadmap (PRM) global path planning with Nonlinear Model Predictive Control (NMPC) for local trajectory tracking. The PRM provides a collision-free reference path, and NMPC optimises tracking while respecting dynamics constraints.

## Key Equations

$$\min_{u_{0:N}} \sum_k \|x_k - x_{\text{ref},k}\|_Q^2 + \|u_k\|_R^2 \quad \text{s.t. } x_{k+1} = f(x_k, u_k), \; u \in \mathcal{U}$$

## Reference

L. Grune, J. Pannek, "Nonlinear Model Predictive Control: Theory and Algorithms," Springer, 2017. [DOI](https://doi.org/10.1007/978-3-319-46024-6)

## Usage

```bash
python -m uav_sim.simulations.trajectory_tracking.nmpc_prm
```

## Result

![nmpc_prm](nmpc_prm.gif)
