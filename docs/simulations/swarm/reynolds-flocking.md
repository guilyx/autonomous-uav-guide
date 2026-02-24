<!-- Erwin Lejeune — 2026-02-24 -->
# Reynolds Flocking

## Problem Statement

Reynolds flocking produces coordinated multi-agent motion from local interaction rules, without centralized planning.

## Model and Formulation

For agent `i`, acceleration is the weighted sum:

$$
a_i = w_s a_i^{sep} + w_a a_i^{align} + w_c a_i^{coh}
$$

where separation avoids collisions, alignment matches heading, and cohesion preserves group compactness.

## Practical Notes

- Perception radius and separation radius define local interaction topology.
- Rule weights set global behavior: tight flocking, milling, or loose travel.
- Speed clipping is essential to prevent unstable divergence.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.reynolds_flocking
```

## Evidence

![Reynolds Flocking](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/reynolds_flocking/reynolds_flocking.gif)

## References

- [Reynolds, Flocks, Herds, and Schools (1987)](https://dl.acm.org/doi/10.1145/37401.37406)
- [Olfati-Saber, Flocking for Multi-Agent Dynamic Systems (2006)](https://doi.org/10.1109/TAC.2005.864190)

## Related Algorithms

- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Potential Swarm](/simulations/swarm/potential-swarm)
