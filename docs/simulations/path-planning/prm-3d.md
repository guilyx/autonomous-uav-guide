<!-- Erwin Lejeune — 2026-02-24 -->
# PRM 3D

## Problem Statement

Probabilistic Roadmaps (PRM) precompute connectivity of free space and then solve fast graph queries between many start-goal pairs.

## Model and Formulation

PRM consists of:

- random free-space samples as graph vertices
- local planner collision checks as graph edges
- shortest-path query over roadmap graph

## Practical Notes

- Useful for multi-query scenarios with static maps.
- Node density controls completeness-quality trade-off.
- Local planner choice strongly influences roadmap connectivity.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_planning.prm_3d
```

## Evidence

![PRM 3D](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/prm_3d/prm_3d.gif)

## References

- [Kavraki et al., Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces (1996)](https://doi.org/10.1109/70.508439)
- [Hsu et al., Path Planning in Expansive Configuration Spaces](https://doi.org/10.1177/02783649922066658)

## Related Algorithms

- [RRT* 3D](/simulations/path-planning/rrt-star-3d)
- [A* 3D](/simulations/path-planning/astar-3d)
