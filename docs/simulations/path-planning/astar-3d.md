<!-- Erwin Lejeune — 2026-02-24 -->
# A* 3D

## Problem Statement

A* provides deterministic, optimal graph search for 3D occupancy grids when an admissible heuristic is available.

## Model and Formulation

Node selection minimizes:

$$
f(n)=g(n)+h(n)
$$

with path cost `g` and heuristic `h` to goal (typically Euclidean distance in 3D).

## Practical Notes

- Heuristic admissibility preserves optimality guarantees.
- Performance depends on branching factor and map resolution.
- Path smoothing is often needed post-search for flyability.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_planning.astar_3d
```

## Evidence

![Astar 3D](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/astar_3d/astar_3d.gif)

## References

- [Hart, Nilsson, Raphael, A Formal Basis for the Heuristic Determination of Minimum Cost Paths (1968)](https://doi.org/10.1109/TSSC.1968.300136)
- [LaValle, Planning Algorithms](http://planning.cs.uiuc.edu/)

## Related Algorithms

- [RRT* 3D](/simulations/path-planning/rrt-star-3d)
- [Costmap Navigation](/simulations/environment/costmap-navigation)
