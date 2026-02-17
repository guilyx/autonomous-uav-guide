# Complementary Filter

## Theory

Fuses high-frequency gyroscope data with low-frequency accelerometer measurements using a tunable parameter $\alpha \in [0,1]$:

$$\hat{\theta}_k = \alpha(\hat{\theta}_{k-1} + \omega_k \Delta t) + (1-\alpha)\theta_{\text{accel},k}$$

Simple, computationally cheap, and well-suited for embedded attitude estimation.

## Reference

R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear Complementary Filters on the Special Orthogonal Group," IEEE TAC, 2008. [DOI: 10.1109/TAC.2008.923738](https://doi.org/10.1109/TAC.2008.923738)

## Simulation

```bash
uv run python simulations/complementary_filter/run.py
```

![Complementary Filter](../../simulations/complementary_filter/complementary_filter.gif)

## API

```python
from quadrotor_sim.estimation.complementary_filter import ComplementaryFilter
cf = ComplementaryFilter(alpha=0.98)
roll, pitch = cf.update(gyro_3, accel_3, dt)
```
