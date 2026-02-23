<!-- Erwin Lejeune — 2026-02-23 -->
# EKF-SLAM

## Algorithm

Simultaneous Localisation and Mapping using an augmented EKF. Landmark positions are appended to the state vector and jointly estimated with the robot pose.

**Reference:** H. Durrant-Whyte, T. Bailey, "Simultaneous Localization and Mapping: Part I," IEEE RAM, 2006.

## Run

```bash
python -m uav_sim.simulations.perception.ekf_slam
```
