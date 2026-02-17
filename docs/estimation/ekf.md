# Extended Kalman Filter (EKF)

## Theory

Linearises a nonlinear system about the current estimate using first-order Taylor expansion (Jacobians):

**Predict:**
$$\hat{x}^- = f(\hat{x}, u), \quad P^- = F P F^\top + Q$$

**Update:**
$$K = P^- H^\top (H P^- H^\top + R)^{-1}$$
$$\hat{x} = \hat{x}^- + K(z - h(\hat{x}^-)), \quad P = (I - KH)P^-$$

## Reference

S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT Press, 2005, Chapter 3.3.

## Simulation

```bash
uv run python simulations/estimation/ekf/run.py
```

![EKF](../../simulations/estimation/ekf/ekf.gif)

## API

```python
from uav_sim.estimation.ekf import ExtendedKalmanFilter
ekf = ExtendedKalmanFilter(state_dim=2, meas_dim=1, f=f, h=h, F_jac=F, H_jac=H)
ekf.predict(u, dt); ekf.update(z)
```
