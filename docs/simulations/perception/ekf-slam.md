<!-- Erwin Lejeune — 2026-02-24 -->
# EKF-SLAM

## Problem Statement

EKF-SLAM jointly estimates vehicle pose and landmark map when both are initially uncertain.
It enables consistent localization in GPS-denied or partially observed environments.

## Model and Formulation

Augmented state:

$$
x = [x_{robot}, m_1, m_2, \dots, m_n]^\top
$$

Prediction/update follow EKF recursion over the full covariance block matrix, preserving robot-landmark cross-correlation.

## Algorithm Procedure

1. Predict robot pose from motion model.
2. Data-associate landmark observations.
3. Update joint state and covariance with measurement Jacobians.
4. Initialize new landmarks when unmatched features appear.

## Tuning and Failure Modes

- Correct data association is critical; outliers can corrupt the entire map.
- Landmark over-parameterization increases computational cost (`O(n^2)` covariance updates).
- Process noise underestimation causes overconfident map geometry.

## Implementation and Execution

```bash
python -m uav_sim.simulations.perception.ekf_slam
```

## Evidence

![EKF SLAM](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/ekf_slam/ekf_slam.gif)

## References

- [Durrant-Whyte and Bailey, Simultaneous Localization and Mapping: Part I](https://doi.org/10.1109/MRA.2006.1638022)
- [Thrun, Burgard, Fox, Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)

## Related Algorithms

- [Occupancy Mapping](/simulations/perception/occupancy-mapping)
- [Extended Kalman Filter](/simulations/estimation/ekf)
