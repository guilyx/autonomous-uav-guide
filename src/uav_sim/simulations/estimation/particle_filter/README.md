# Particle Filter

A sequential Monte Carlo method that represents the belief distribution as a set of weighted particles. Handles nonlinear, non-Gaussian estimation by sampling, weighting by measurement likelihood, and resampling.

## Key Equations

$$w_k^{(i)} \propto p(z_k \mid x_k^{(i)}), \quad \hat x_k = \sum_i w_k^{(i)} x_k^{(i)}$$

## Reference

M. S. Arulampalam et al., "A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking," IEEE Trans. SP, 2002. [DOI](https://doi.org/10.1109/78.978374)

## Usage

```bash
python -m uav_sim.simulations.estimation.particle_filter
```

## Result

![particle_filter](particle_filter.gif)
