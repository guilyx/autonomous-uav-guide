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

- **Beam angles are body-referenced; the vehicle heading has to go in
  too.** Integrating every scan as though the vehicle pointed along `+x`
  produces a map that is correct only while it flies straight and smears
  progressively as it turns.
- **Treat a range near the sensor maximum as a no-return.** Range noise
  pushes a genuine miss just under the limit, and marking that cell
  occupied paints a phantom obstacle at the edge of every scan — a ring of
  false wall around the vehicle. Gate the occupied update with a margin.
- Log-odds must be clipped, or the sigmoid overflows once a cell has been
  seen enough times.

- Incorrect sensor model causes inflated false positives/negatives.
- Dynamic obstacles can leave ghost occupancy without decay logic.
- Grid resolution too coarse obscures narrow passages.

## Implementation and Execution

```bash
python -m uav_sim.simulations.perception.occupancy_mapping
```

## Evidence

![Occupancy Mapping](https://media.githubusercontent.com/media/guilyx/flybots/main/src/uav_sim/simulations/perception/occupancy_mapping/occupancy_mapping.gif)

## References

- [Thrun, Burgard, Fox, Probabilistic Robotics](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)
- [Elfes, Occupancy Grids for Mobile Robot Perception and Navigation (1989)](https://doi.org/10.1109/ROBOT.1989.100285)

## Related Algorithms

- [Costmap Navigation](/simulations/environment/costmap-navigation)
- [A* 3D](/simulations/path-planning/astar-3d)
