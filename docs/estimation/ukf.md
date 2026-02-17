# Unscented Kalman Filter (UKF)

## Theory

Uses deterministic sigma points to propagate uncertainty through nonlinear functions, avoiding Jacobian computation. The Van der Merwe scaled unscented transform generates $2n+1$ sigma points:

$$\mathcal{X}_0 = \hat{x}, \quad \mathcal{X}_i = \hat{x} \pm \sqrt{(n+\lambda)P}$$

These are propagated through the dynamics/measurement model and the posterior mean and covariance are recovered from weighted combinations.

## Reference

E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for Nonlinear Estimation," AS-SPCC, 2000. [DOI: 10.1109/ASSPCC.2000.882463](https://doi.org/10.1109/ASSPCC.2000.882463)

## Simulation

```bash
uv run python simulations/ukf/run.py
```

![UKF](../../simulations/ukf/ukf.gif)

## API

```python
from quadrotor_sim.estimation.ukf import UnscentedKalmanFilter
ukf = UnscentedKalmanFilter(state_dim=2, meas_dim=1, f=f, h=h)
ukf.predict(u, dt); ukf.update(z)
```
