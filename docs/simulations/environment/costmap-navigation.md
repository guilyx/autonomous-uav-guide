<!-- Erwin Lejeune — 2026-02-24 -->
# Costmap Navigation

## Problem Statement

Costmap navigation fuses occupancy information into a traversability surface used for local replanning and obstacle avoidance.
It bridges mapping uncertainty with actionable path costs.

## Model and Formulation

A costmap cell combines occupancy, inflation, and dynamic penalties:

$$
C(x,y)=w_o C_{occ}(x,y)+w_i C_{infl}(x,y)+w_d C_{dyn}(x,y)
$$

Local planning repeatedly solves shortest-path queries on the evolving cost field.

## Algorithm Procedure

1. Update occupancy with sensor observations.
2. Apply footprint inflation and dynamic obstacle penalties.
3. Replan local route at fixed frequency.
4. Feed feasible path segments to tracking controller.

## Tuning and Failure Modes

- Over-inflation can block narrow but valid corridors.
- Under-inflation reduces safety margins near moving obstacles.
- Replan frequencies that are too low produce stale avoidance behavior.

## Implementation and Execution

```bash
python -m uav_sim.simulations.environment.costmap_navigation
```

## Evidence

![Costmap Navigation](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/environment/costmap_navigation/costmap_navigation.gif)

## References

- [Fox, Burgard, Thrun, Dynamic Window Approach (1997)](https://doi.org/10.1109/100.580977)
- [Lu et al., Layered Costmaps for Context-Sensitive Navigation (IROS 2014)](https://doi.org/10.1109/IROS.2014.6942367)

## Related Algorithms

- [Occupancy Mapping](/simulations/perception/occupancy-mapping)
- [A* 3D](/simulations/path-planning/astar-3d)
