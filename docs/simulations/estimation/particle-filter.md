<!-- Erwin Lejeune — 2026-02-23 -->
# Particle Filter

## Algorithm

Represents the posterior distribution as a set of weighted particles. Uses Sequential Importance Resampling (SIR) with systematic resampling.

**Reference:** M. S. Arulampalam et al., "A Tutorial on Particle Filters," IEEE TSP, 2002.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N particles | 200 | Particle count |
| GPS std | 0.6 m | Measurement noise |
| Resample threshold | N_eff < N/2 | Effective sample size |

## Run

```bash
python -m uav_sim.simulations.estimation.particle_filter
```
