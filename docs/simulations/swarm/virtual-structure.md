<!-- Erwin Lejeune — 2026-02-24 -->
# Virtual Structure Formation

## Problem Statement

Virtual-structure control treats the swarm as one rigid body with assigned slots for each agent.
It enables precise geometric formations during coordinated maneuvers.

## Model and Formulation

Each agent tracks:

$$
p_i^{ref}(t) = p_{vs}(t) + R_{vs}(t) r_i
$$

where `p_{vs}, R_{vs}` define the structure pose and `r_i` is agent slot offset.

## Practical Notes

- Works best with reliable relative localization.
- Slot assignment should minimize crossing paths during reconfiguration.
- Tracking gains should account for heterogeneous agent dynamics.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.virtual_structure
```

## Evidence

![Virtual Structure](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/virtual_structure/virtual_structure.gif)

## References

- [Lewis and Tan, High Precision Formation Control Using Virtual Structures (1997)](https://doi.org/10.1023/A:1008814708459)
- [Beard et al., Coordination Variables and Consensus Building in Multiple Vehicle Systems](https://doi.org/10.1002/rob.20127)

## Related Algorithms

- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Leader-Follower](/simulations/swarm/leader-follower)
