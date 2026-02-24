<!-- Erwin Lejeune — 2026-02-23 -->
# Feedback Linearisation

## Algorithm

Exploits the differential flatness of the quadrotor to invert the nonlinear dynamics, reducing trajectory tracking to linear error dynamics with PD gains on position error.

**Reference:** D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors," ICRA, 2011, Sec. IV.

## Run

```bash
python -m uav_sim.simulations.trajectory_tracking.feedback_linearisation
```
