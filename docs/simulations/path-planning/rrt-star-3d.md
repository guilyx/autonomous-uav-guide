<!-- Erwin Lejeune — 2026-02-24 -->
# RRT* 3D

## Problem Statement

RRT* incrementally builds a tree in continuous space and asymptotically improves path cost through rewiring.

## Model and Formulation

At each iteration:

1. Sample random state `x_rand`
2. Extend nearest node toward sample
3. Choose parent minimizing local cost
4. Rewire neighbors if new path improves their cost

As iterations grow, solution cost converges toward optimal.

## Practical Notes

- Goal bias improves convergence speed.
- Step size controls exploration granularity.
- Collision checks dominate runtime.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_planning.rrt_star_3d
```

## Evidence

![RRT Star 3D](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/rrt_star_3d/rrt_star_3d.gif)

## References

- [Karaman and Frazzoli, Sampling-based Algorithms for Optimal Motion Planning (2011)](https://doi.org/10.1177/0278364911406761)
- [Lavalle and Kuffner, Rapidly-exploring Random Trees](https://doi.org/10.1177/02783640122067453)

## Related Algorithms

- [PRM 3D](/simulations/path-planning/prm-3d)
- [A* 3D](/simulations/path-planning/astar-3d)
