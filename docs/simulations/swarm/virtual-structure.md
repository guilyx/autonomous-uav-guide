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

- Feed the **slot velocity** forward, not just its position. Damping
  against the world frame instead of against the moving slot leaves a
  standing formation error of `k_d·v_body / k_p` — proportional to how
  fast the structure travels, and easy to mistake for a gain that needs
  raising. With the feed-forward wired up, formation error on the atlas
  demo drops from 2.11 m to 0.16 m.
- For a rotating structure the slot velocity includes the tangential
  `ω × r` term, and the slot acceleration the centripetal `ω × (ω × r)`.
  A rigid body that yaws with its path needs both.
- Do not stack a multiplicative velocity decay on top of the PD's own
  `k_d`. The decay behaves as unmodelled drag: it demands a steady force
  to hold a steady velocity, which the PD can only produce from standing
  error. The agent model here is a plain double integrator.

- Works best with reliable relative localization.
- Slot assignment should minimize crossing paths during reconfiguration.
- Tracking gains should account for heterogeneous agent dynamics.

## Implementation and Execution

```bash
python -m flybots.simulations.swarm.virtual_structure
```

## Evidence

![Virtual Structure](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/swarm/virtual_structure/virtual_structure.gif)

## References

- [Lewis and Tan, High Precision Formation Control Using Virtual Structures (1997)](https://doi.org/10.1023/A:1008814708459)
- [Beard et al., Coordination Variables and Consensus Building in Multiple Vehicle Systems](https://doi.org/10.1002/rob.20127)

## Related Algorithms

- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Leader-Follower](/simulations/swarm/leader-follower)
