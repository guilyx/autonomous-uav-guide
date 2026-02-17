# 3D A* Path Planner

## Theory

Extends the classic A* graph search to a 3D voxel grid. Uses an admissible heuristic (Euclidean distance) to guarantee optimality in the discretised space. The algorithm maintains an open set (priority queue) ordered by $f(n) = g(n) + h(n)$ and expands 26-connected neighbours.

## Reference

P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths," IEEE TSSC, 1968. [DOI: 10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136)

## Simulation

```bash
uv run python simulations/path_planning/astar_3d/run.py
```

![A* 3D](../../simulations/path_planning/astar_3d/astar_3d.gif)

## API

```python
from uav_sim.path_planning.astar_3d import AStar3D
planner = AStar3D(occupancy_grid)
path = planner.plan(start_tuple, goal_tuple)
```
