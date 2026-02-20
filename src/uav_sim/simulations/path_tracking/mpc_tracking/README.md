# Model Predictive Control (MPC) Tracking

Solves a finite-horizon optimal control problem at each step to track a reference path. The receding-horizon approach naturally handles constraints on thrust and tilt angle while optimising tracking performance.

## Key Equations

$$\min_{u_{0:N-1}} \sum_{k=0}^{N-1} \|x_k - x_{\text{ref}}\|_Q^2 + \|u_k\|_R^2 \quad \text{s.t. } x_{k+1} = f(x_k, u_k)$$

## Reference

M. Kamel et al., "Model Predictive Control for Trajectory Tracking of Unmanned Aerial Vehicles Using Robot Operating System," Springer, 2017. [DOI](https://doi.org/10.1007/978-3-319-54927-9_1)

## Usage

```bash
python -m uav_sim.simulations.path_tracking.mpc_tracking
```

## Result

![mpc_tracking](mpc_tracking.gif)
