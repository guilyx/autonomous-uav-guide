<!-- Erwin Lejeune — 2026-02-23 -->
# Model Predictive Path Integral (MPPI)

## Algorithm

Sampling-based trajectory optimiser that draws K random control sequences, evaluates their cost, and computes an information-theoretic weighted average. Outputs wrench commands directly.

**Reference:** G. Williams et al., "Information Theoretic MPC for Model-Based Reinforcement Learning," ICRA, 2017.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Horizon | 25 | Prediction steps |
| Samples K | 256 | Random rollouts |
| λ | 0.5 | Temperature |
| Control dim | 4 | [T, τx, τy, τz] |

## Run

```bash
python -m uav_sim.simulations.trajectory_tracking.mppi
```
