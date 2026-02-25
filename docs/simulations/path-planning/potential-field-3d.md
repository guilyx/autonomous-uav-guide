<!-- Erwin Lejeune — 2026-02-24 -->
# Potential Field 3D

## Problem Statement

Potential-field planning computes motion by descending a synthetic scalar field that attracts toward goals and repels from obstacles.

## Model and Formulation

Total potential:

$$
U(q)=U_{att}(q)+U_{rep}(q)
$$

Motion command:

$$
\dot{q}=-\nabla U(q)
$$

## Practical Notes

- Very fast and simple for local navigation.
- Sensitive to local minima in cluttered environments.
- Often combined with global planners for robust deployment.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_planning.potential_field_3d
```

## Evidence

![Potential Field 3D](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_planning/potential_field_3d/potential_field_3d.gif)

## References

- [Khatib, Real-Time Obstacle Avoidance for Manipulators and Mobile Robots (1986)](https://doi.org/10.1177/027836498600500106)
- [Ge and Cui, Dynamic Motion Planning for Mobile Robots Using Potential Field Method](https://doi.org/10.1007/3-540-44514-9_31)

## Related Algorithms

- [A* 3D](/simulations/path-planning/astar-3d)
- [Costmap Navigation](/simulations/environment/costmap-navigation)
