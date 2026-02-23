<!-- Erwin Lejeune — 2026-02-23 -->
# Nonlinear MPC (NMPC)

## Algorithm

Receding-horizon optimisation using single-shooting with RK2 integration. Solves a constrained nonlinear program at each control step to track position and velocity references.

**Reference:** M. Diehl et al., "Real-Time Optimization and Nonlinear Model Predictive Control of Processes Governed by Differential-Algebraic Equations," J. Process Control, 2002.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Horizon | 15 | Prediction steps |
| dt_ctrl | 0.02 s | Control rate (50 Hz) |
| dt_sim | 0.005 s | Physics rate (200 Hz) |

## Run

```bash
python -m uav_sim.simulations.trajectory_tracking.nmpc
```
