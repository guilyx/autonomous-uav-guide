# Voronoi-Based Area Coverage

## Theory

Lloyd's algorithm drives each agent towards the centroid of its Voronoi cell, achieving optimal area coverage. At each step:

1. Compute the Voronoi partition of the workspace.
2. Find the centroid of each cell (weighted by a density function).
3. Move each agent towards its centroid.

Convergence is guaranteed for convex workspaces.

## Reference

J. Cortes, S. Martinez, T. Karatas, F. Bullo, "Coverage Control for Mobile Sensing Networks," IEEE T-RA, 2004. [DOI: 10.1109/TRA.2004.824698](https://doi.org/10.1109/TRA.2004.824698)

## Simulation

```bash
uv run python simulations/swarm/voronoi_coverage/run.py
```

![Voronoi Coverage](../../simulations/swarm/voronoi_coverage/voronoi_coverage.gif)

## API

```python
from uav_sim.swarm.coverage import CoverageController
ctrl = CoverageController(bounds=np.array([[0,10],[0,10]]))
forces = ctrl.compute_forces(positions_2d)
```
