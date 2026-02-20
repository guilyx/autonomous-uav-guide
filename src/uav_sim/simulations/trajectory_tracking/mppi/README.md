# Model Predictive Path Integral (MPPI)

A sampling-based trajectory optimisation method that rolls out many noisy trajectories through the dynamics model and weights them by their cost. The optimal control is a cost-weighted average over all rollouts. Handles nonlinear dynamics and non-convex costs naturally.

## Key Equations

$$u^* = \sum_i w_i\,\epsilon_i, \quad w_i = \frac{\exp(-\frac{1}{\lambda}S_i)}{\sum_j \exp(-\frac{1}{\lambda}S_j)}$$

## Reference

G. Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning," ICRA 2017. [DOI](https://doi.org/10.1109/ICRA.2017.7989202)

## Usage

```bash
python -m uav_sim.simulations.trajectory_tracking.mppi
```

## Result

![mppi](mppi.gif)
