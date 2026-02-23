<!-- Erwin Lejeune — 2026-02-23 -->
# Voronoi Coverage (Lloyd's Algorithm)

## Algorithm

Agents iteratively move to the centroids of their Voronoi cells, converging to an optimal coverage configuration that minimises the maximum distance from any point to the nearest agent.

**Reference:** J. Cortes et al., "Coverage Control for Mobile Sensing Networks," IEEE T-RA, 2004.

## Run

```bash
python -m uav_sim.simulations.swarm.voronoi_coverage
```
