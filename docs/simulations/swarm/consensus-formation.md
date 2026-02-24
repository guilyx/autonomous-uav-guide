<!-- Erwin Lejeune — 2026-02-24 -->
# Consensus Formation

## Problem Statement

Consensus formation seeks distributed agreement on relative geometry using only neighbor communication.

## Model and Formulation

Continuous consensus protocol:

$$
\dot{x}_i = -\sum_{j \in \mathcal{N}_i} a_{ij}(x_i - x_j - \Delta_{ij})
$$

where `\Delta_{ij}` encodes desired formation offsets.

## Practical Notes

- Graph connectivity is a hard requirement for convergence.
- Consensus gain affects settling time versus oscillation.
- Communication delays can degrade phase alignment.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.consensus_formation
```

## Evidence

![Consensus Formation](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/consensus_formation/consensus_formation.gif)

## References

- [Olfati-Saber and Murray, Consensus Problems in Networks of Agents (2004)](https://doi.org/10.1109/TAC.2004.834113)
- [Ren and Beard, Consensus Seeking in Multi-Agent Systems](https://doi.org/10.1007/978-1-84628-981-1)

## Related Algorithms

- [Virtual Structure](/simulations/swarm/virtual-structure)
- [Leader-Follower](/simulations/swarm/leader-follower)
