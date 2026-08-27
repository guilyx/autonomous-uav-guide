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

Reynolds' fourth term, the **migratory urge**, is what makes the flock go
somewhere:

$$
a_i \mathrel{+}= w_m\left(v^{\text{cruise}} - v_i\right)
$$

## Practical Notes

- Perception radius and separation radius define local interaction topology.
- Rule weights set global behavior: tight flocking, milling, or loose travel.
- Speed clipping is essential to prevent unstable divergence.
- **The three classic rules all go to zero once the flock is in
  formation.** Separation, alignment and cohesion describe how agents
  arrange themselves relative to each other, not where the group goes, so
  with any velocity damping a flock running on those three alone
  coasts to a standstill and sits there — arranged correctly, going
  nowhere. The migratory urge is not decoration.
- Perception radius has to cover the initial spread. An agent that starts
  outside everyone's radius has no neighbours, and nothing in the local
  rules will ever bring it back.

## Implementation and Execution

```bash
python -m flybots.simulations.swarm.reynolds_flocking
```

## Evidence

![Reynolds Flocking](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/swarm/reynolds_flocking/reynolds_flocking.gif)

## References

- [Reynolds, Flocks, Herds, and Schools (1987)](https://dl.acm.org/doi/10.1145/37401.37406)
- [Olfati-Saber, Flocking for Multi-Agent Dynamic Systems (2006)](https://doi.org/10.1109/TAC.2005.864190)

## Related Algorithms

- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Potential Swarm](/simulations/swarm/potential-swarm)
