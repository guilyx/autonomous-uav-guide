<!-- Erwin Lejeune — 2026-02-24 -->
# Coverage Planning

## Problem Statement

Coverage planning generates paths that guarantee spatial visitation for inspection, mapping, or search missions.

## Model and Formulation

Coverage objective:

$$
\max \int_{\Omega} \mathbb{1}_{visited}(q)\,dq
$$

subject to vehicle dynamics, footprint size, and mission-time constraints.

## Practical Notes

- Common decompositions include boustrophedon and grid sweep methods.
- Turn penalties should be considered for fixed-wing platforms.
- Map uncertainty can trigger adaptive revisitation.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_planning.coverage_planning
```

## Evidence

![Coverage Planning](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/coverage_planning/coverage_planning.gif)

## References

- [Choset, Coverage for Robotics: A Survey (2001)](https://doi.org/10.1016/S0921-8890(00)00078-7)
- [Galceran and Carreras, A Survey on Coverage Path Planning for Robotics (2013)](https://doi.org/10.1016/j.robot.2012.09.004)

## Related Algorithms

- [Voronoi Coverage](/simulations/swarm/voronoi-coverage)
- [PRM 3D](/simulations/path-planning/prm-3d)
