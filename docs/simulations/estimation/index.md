<!-- Erwin Lejeune — 2026-02-24 -->
# Estimation

Estimation reconstructs latent vehicle state from noisy, asynchronous sensors under model uncertainty.
This chapter focuses on Bayesian and complementary estimators used by downstream control and planning layers.

## Core Questions

- How should process and measurement uncertainty be represented?
- When does linearization error dominate filter performance?
- What is the observability footprint of each sensor suite?

## Algorithms

- [Complementary Filter](/simulations/estimation/complementary-filter)
- [Extended Kalman Filter](/simulations/estimation/ekf)
- [Unscented Kalman Filter](/simulations/estimation/ukf)
- [GPS-IMU Fusion](/simulations/estimation/gps-imu-fusion)
- [Particle Filter](/simulations/estimation/particle-filter)

## Prerequisites

- Continuous-time rigid body dynamics
- Gaussian estimation and covariance propagation
- IMU and GPS measurement models
