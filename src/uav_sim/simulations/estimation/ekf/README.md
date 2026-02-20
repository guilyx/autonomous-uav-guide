# Extended Kalman Filter (EKF)

Estimates the 3D position and velocity of a quadrotor from noisy GPS and IMU measurements. The EKF linearises the nonlinear process model around the current estimate at each step, balancing prediction uncertainty against measurement noise.

## Key Equations

$$\hat x_{k|k} = \hat x_{k|k-1} + K_k\,(z_k - h(\hat x_{k|k-1})), \quad K_k = P_{k|k-1}H_k^T(H_kP_{k|k-1}H_k^T + R)^{-1}$$

## Reference

G. Welch, G. Bishop, "An Introduction to the Kalman Filter," UNC Chapel Hill TR 95-041, 2006. [Link](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)

## Usage

```bash
python -m uav_sim.simulations.estimation.ekf
```

## Result

![ekf](ekf.gif)
