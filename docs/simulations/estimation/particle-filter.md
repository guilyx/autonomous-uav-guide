<!-- Erwin Lejeune — 2026-02-24 -->
# Particle Filter

## Problem Statement

When posterior distributions are multi-modal or strongly non-Gaussian, Kalman-family filters become brittle.
Particle filters approximate the full posterior with weighted samples and remain effective in nonlinear, non-Gaussian settings.

## Model and Formulation

The posterior is approximated by particles:

$$
p(x_k|z_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)}\delta\left(x_k - x_k^{(i)}\right)
$$

Weights are updated from measurement likelihood:

$$
w_k^{(i)} \propto w_{k-1}^{(i)}p(z_k|x_k^{(i)})
$$

Resampling is triggered by effective sample size:

$$
N_{eff} = \frac{1}{\sum_i (w_k^{(i)})^2}
$$

## Algorithm Procedure

1. Sample particles from transition model.
2. Evaluate measurement likelihood for each particle.
3. Normalize weights and compute `N_eff`.
4. Resample when degeneracy threshold is crossed.

## Tuning Guidance

- Increase particle count for higher-dimensional states.
- Match proposal noise to platform maneuver envelope.
- Use stratified/systematic resampling to reduce variance.

## Failure Modes and Diagnostics

- Particle impoverishment occurs with frequent resampling and narrow proposals.
- Sparse particle sets miss low-probability but valid hypotheses.
- Computational load scales with particle count and measurement complexity.

## Implementation and Execution

```bash
python -m uav_sim.simulations.estimation.particle_filter
```

## Evidence

![Particle Filter](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/estimation/particle_filter/particle_filter.gif)

## References

- [Arulampalam et al., A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking (2002)](https://doi.org/10.1109/78.978374)
- [Doucet and Johansen, A Tutorial on Particle Filtering and Smoothing (2009)](https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf)

## Related Algorithms

- [Unscented Kalman Filter](/simulations/estimation/ukf)
- [EKF-SLAM](/simulations/perception/ekf-slam)
