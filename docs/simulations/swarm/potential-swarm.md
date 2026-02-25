<!-- Erwin Lejeune — 2026-02-24 -->
# Potential-Based Swarm

## Problem Statement

Potential-field swarm control combines attractive and repulsive fields to produce distributed collision-avoiding collective motion.

## Model and Formulation

Agent force model:

$$
F_i = -\nabla U_{goal}(p_i) - \sum_{j \ne i}\nabla U_{ij}(p_i,p_j)
$$

where `U_{ij}` can be Lennard-Jones-like or quadratic barrier potentials.

## Practical Notes

- Potential shape determines spacing and rigidity.
- Local minima are a known issue in cluttered environments.
- Add damping terms to prevent oscillatory interactions.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.potential_swarm
```

## Evidence

![Potential Swarm](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/potential_swarm/potential_swarm.gif)

## References

- [Spears et al., Distributed Physics-Based Control of Swarms (2004)](https://doi.org/10.1023/B:AURO.0000033970.96785.f1)
- [Khatib, Real-Time Obstacle Avoidance for Manipulators and Mobile Robots (1986)](https://doi.org/10.1177/027836498600500106)

## Related Algorithms

- [Reynolds Flocking](/simulations/swarm/reynolds-flocking)
- [Voronoi Coverage](/simulations/swarm/voronoi-coverage)
