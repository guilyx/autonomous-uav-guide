# Particle Filter

## Theory

A sequential Monte Carlo method that represents the posterior distribution as a set of weighted particles. At each step:

1. **Predict**: propagate each particle through the dynamics model with process noise.
2. **Update**: weight each particle by its likelihood given the measurement.
3. **Resample**: draw new particles proportional to weights (systematic resampling).

Handles multi-modal and highly nonlinear distributions where Kalman-based filters struggle.

## Reference

M. S. Arulampalam, S. Maskell, N. Gordon, T. Clapp, "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking," IEEE TSP, 2002. [DOI: 10.1109/78.978374](https://doi.org/10.1109/78.978374)

## Simulation

```bash
uv run python simulations/estimation/particle_filter/run.py
```

![Particle Filter](../../simulations/estimation/particle_filter/particle_filter.gif)

## API

```python
from uav_sim.estimation.particle_filter import ParticleFilter
pf = ParticleFilter(state_dim=2, num_particles=300, f=f, likelihood=lik)
pf.predict(u, dt); pf.update(z)
estimate = pf.estimate
```
