# Unscented Kalman Filter (UKF)

An alternative to the EKF that avoids explicit Jacobian computation. Deterministic sigma points are propagated through the nonlinear model to capture the posterior mean and covariance to third-order accuracy.

Here the drone has no GPS. It localises from noisy **ranges to four
surveyed ground anchors** — a measurement model that is nonlinear in the
state, which is what makes this a UKF demo rather than a Kalman-filter
demo. Given a linear model the UKF reduces exactly to the KF and there
would be nothing to show.

The process model stays a constant-velocity coast, so the filter's `Q`
is built from an acceleration power spectral density with
`constant_velocity_q` rather than a hand-written `diag()`. `Q` is the
covariance accumulated over **one step**: a fixed diagonal at 200 Hz
tells the filter its own prediction is worthless, and it collapses onto
the raw measurement.

## Key Equations

$$\mathcal{X}_i = \hat x \pm \sqrt{(n+\lambda)P}, \quad \hat x^- = \sum_i W_i^m\,f(\mathcal{X}_i)$$

$$h(x) = \big[\,\lVert p - a_1 \rVert,\ \dots,\ \lVert p - a_4 \rVert\,\big]$$

## Reference

S. J. Julier, J. K. Uhlmann, "Unscented Filtering and Nonlinear Estimation," Proc. IEEE, 2004. [DOI](https://doi.org/10.1109/JPROC.2003.823141)

## Usage

```bash
python -m uav_sim.simulations.estimation.ukf
```

## Result

![ukf](ukf.gif)
