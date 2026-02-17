# Model Predictive Path Integral (MPPI)

## Theory

A sampling-based MPC method that draws random control perturbations, rolls out the dynamics for each sample, evaluates costs, and computes the optimal control as an importance-weighted average:

$$u^*_t = u_t + \frac{\sum_k w_k \epsilon_{k,t}}{\sum_k w_k}, \quad w_k = \exp\left(-\frac{1}{\lambda}S_k\right)$$

where $S_k$ is the total trajectory cost and $\lambda$ is a temperature parameter.

## Reference

G. Williams, N. Wagener, B. Goldfain, P. Drews, J. M. Rehg, B. Boots, E. A. Theodorou, "Information Theoretic MPC for Model-Based Reinforcement Learning," ICRA, 2017. [DOI: 10.1109/ICRA.2017.7989202](https://doi.org/10.1109/ICRA.2017.7989202)

## Simulation

```bash
uv run python simulations/mppi/run.py
```

![MPPI](../../simulations/mppi/mppi.gif)

## API

```python
from quadrotor_sim.tracking.mppi import MPPITracker
tracker = MPPITracker(state_dim=6, control_dim=3, dynamics=dyn, cost_fn=cost)
u = tracker.compute(state, reference=target)
```
