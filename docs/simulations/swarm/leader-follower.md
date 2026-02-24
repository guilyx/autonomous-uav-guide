<!-- Erwin Lejeune — 2026-02-24 -->
# Leader-Follower

## Problem Statement

Leader-follower architectures simplify coordination by assigning one trajectory source and distributed tracking controllers for followers.

## Model and Formulation

Follower error in leader frame:

$$
e_i = p_i - (p_L + \Delta_i)
$$

Control law:

$$
u_i = -K_p e_i - K_d \dot{e}_i
$$

## Practical Notes

- Robust leader estimation is critical for formation stability.
- Follower chain topologies are simple but vulnerable to upstream failure.
- Relative-sensing noise can cause accordion oscillations.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.leader_follower
```

## Evidence

![Leader Follower](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/leader_follower/leader_follower.gif)

## References

- [Desai et al., Modeling and Control of Formations of Nonholonomic Mobile Robots (2001)](https://doi.org/10.1109/70.938381)
- [Oh, Park, Ahn, Survey of Multi-Agent Formation Control (2015)](https://doi.org/10.1016/j.automatica.2014.10.022)

## Related Algorithms

- [Virtual Structure](/simulations/swarm/virtual-structure)
- [Consensus Formation](/simulations/swarm/consensus-formation)
