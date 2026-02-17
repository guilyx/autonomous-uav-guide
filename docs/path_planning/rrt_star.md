# RRT / RRT* Sampling-Based Planner

## Theory

**RRT** grows a tree by repeatedly sampling random configurations and extending the nearest node. It rapidly explores the state space but does not guarantee optimal paths.

**RRT\*** adds a *rewire* step: after extending, it searches nearby nodes for a lower-cost parent, and re-parents neighbours if the new node provides a shortcut. This yields asymptotic optimality.

Near radius: $r = \gamma \left(\frac{\log n}{n}\right)^{1/3}$

## References

1. S. M. LaValle, "Rapidly-Exploring Random Trees: A New Tool for Path Planning," TR 98-11, 1998.
2. S. Karaman, E. Frazzoli, "Sampling-based Algorithms for Optimal Motion Planning," IJRR, 2011. [DOI: 10.1177/0278364911406761](https://doi.org/10.1177/0278364911406761)

## Simulation

```bash
uv run python simulations/path_planning/rrt_star_3d/run.py
```

![RRT* 3D](../../simulations/path_planning/rrt_star_3d/rrt_star_3d.gif)

## API

```python
from uav_sim.path_planning.rrt_3d import RRTStar3D
planner = RRTStar3D(bounds_min, bounds_max, obstacles, step_size=1.0)
path = planner.plan(start, goal)
```
