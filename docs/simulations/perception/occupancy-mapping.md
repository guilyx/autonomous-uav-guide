<!-- Erwin Lejeune — 2026-02-24 -->
# Occupancy Mapping

## Problem Statement

Occupancy mapping converts range observations into a probabilistic spatial model for collision checking and navigation.
It underpins path planning and local obstacle avoidance.

## Model and Formulation

Log-odds update for each cell:

$$
L_t(m_i)=L_{t-1}(m_i)+\log\frac{p(m_i|z_t)}{1-p(m_i|z_t)}-L_0
$$

Probability recovery:

$$
p(m_i)=1-\frac{1}{1+\exp(L_t(m_i))}
$$

## Algorithm Procedure

1. Ray-cast each lidar measurement through the grid.
2. Mark traversed cells as free and endpoint as occupied.
3. Update log-odds with inverse sensor model.
4. Export occupancy map to planning modules.

## Tuning and Failure Modes

- Incorrect sensor model causes inflated false positives/negatives.
- Dynamic obstacles can leave ghost occupancy without decay logic.
- Grid resolution too coarse obscures narrow passages.

## Implementation and Execution

```bash
python -m uav_sim.simulations.perception.occupancy_mapping
```

## Evidence

![Occupancy Mapping](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/perception/occupancy_mapping/occupancy_mapping.gif)

## References

- [Thrun, Burgard, Fox, Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)
- [Elfes, Occupancy Grids for Mobile Robot Perception and Navigation (1989)](https://doi.org/10.1109/ROBOT.1989.100285)

## Related Algorithms

- [Costmap Navigation](/simulations/environment/costmap-navigation)
- [A* 3D](/simulations/path-planning/astar-3d)
