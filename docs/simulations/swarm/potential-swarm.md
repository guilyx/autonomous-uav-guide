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

- **Saturate the goal attraction.** Unsaturated linear attraction is
  thousands of times stronger than the lattice forces at the start of a
  long transit, so the swarm crosses the map as a disordered blob and only
  forms up on arrival. Capping it at a fixed radius keeps formation-keeping
  and navigation comparable throughout.
- **Clamp the obstacle repulsion.** The `1/d²` term is unbounded, so an
  agent that clips an obstacle receives an infinite kick and leaves the
  world. Floor the surface distance and cap the total force.
- **Score the centroid, not the agents.** A lattice converges with each
  agent one spacing away from the goal by construction, so mean
  agent-to-goal distance never goes to zero and reads like a failure. The
  meaningful numbers are centroid-to-goal error and nearest-neighbour
  spacing against `d_des`.

- Potential shape determines spacing and rigidity.
- Local minima are a known issue in cluttered environments.
- Add damping terms to prevent oscillatory interactions.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.potential_swarm
```

## Evidence

![Potential Swarm](https://media.githubusercontent.com/media/guilyx/flybots/main/src/uav_sim/simulations/swarm/potential_swarm/potential_swarm.gif)

## References

- [Spears et al., Distributed Physics-Based Control of Swarms (2004)](https://doi.org/10.1023/B:AURO.0000033970.96785.f1)
- [Khatib, Real-Time Obstacle Avoidance for Manipulators and Mobile Robots (1986)](https://doi.org/10.1177/027836498600500106)

## Related Algorithms

- [Reynolds Flocking](/simulations/swarm/reynolds-flocking)
- [Voronoi Coverage](/simulations/swarm/voronoi-coverage)
