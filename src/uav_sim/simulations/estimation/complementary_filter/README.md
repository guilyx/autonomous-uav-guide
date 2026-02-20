# Complementary Filter

Fuses accelerometer and gyroscope readings to estimate roll and pitch angles. A tunable parameter alpha blends the high-frequency gyro integration with the low-frequency accelerometer reference, acting as a first-order high-pass / low-pass pair.

## Key Equations

$$\hat\theta_k = \alpha\,(\hat\theta_{k-1} + \omega\,\Delta t) + (1-\alpha)\,\theta_{\text{accel}}$$

## Reference

R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear Complementary Filters on the Special Orthogonal Group," IEEE TAC, 2008. [DOI](https://doi.org/10.1109/TAC.2008.919528)

## Usage

```bash
python -m uav_sim.simulations.estimation.complementary_filter
```

## Result

![complementary_filter](complementary_filter.gif)
