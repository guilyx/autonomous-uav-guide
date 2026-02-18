# Unscented Kalman Filter (UKF)

An alternative to the EKF that avoids explicit Jacobian computation. Deterministic sigma points are propagated through the nonlinear model to capture the posterior mean and covariance to third-order accuracy.

## Key Equations

$$\mathcal{X}_i = \hat x \pm \sqrt{(n+\lambda)P}, \quad \hat x^- = \sum_i W_i^m\,f(\mathcal{X}_i)$$

## Reference

S. J. Julier, J. K. Uhlmann, "Unscented Filtering and Nonlinear Estimation," Proc. IEEE, 2004. [DOI](https://doi.org/10.1109/JPROC.2003.823141)

## Usage

```bash
python -m uav_sim.simulations.estimation.ukf
```

## Result

![ukf](ukf.gif)
