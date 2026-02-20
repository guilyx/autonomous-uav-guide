# Nonlinear Model Predictive Control (NMPC)

Online trajectory tracking using single-shooting NMPC with RK2 integration. The controller runs at 50 Hz, receiving look-ahead targets from a pure-pursuit tracker on a global path and optimising a short horizon trajectory using L-BFGS-B.

## Key Equations

$$\min_{u_{0:N}} \sum_k \|x_k - x_{\text{ref},k}\|_Q^2 + \|u_k\|_R^2 \quad \text{s.t. } x_{k+1} = f(x_k, u_k), \; u \in \mathcal{U}$$

## Reference

M. Diehl et al., "Real-Time Optimization and Nonlinear Model Predictive Control," J. Process Control, 2002. [DOI](https://doi.org/10.1016/S0959-1524(02)00023-1)

## Usage

```bash
python -m uav_sim.simulations.trajectory_tracking.nmpc
```

## Result

![nmpc](nmpc.gif)
