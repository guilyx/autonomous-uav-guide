<!-- Erwin Lejeune — 2026-02-24 -->
# Voronoi Coverage (Lloyd's Algorithm)

## Problem Statement

Coverage control places agents to minimize sensing distance over an area.
Voronoi partitioning with Lloyd descent yields distributed, geometrically interpretable coverage behavior.

## Model and Formulation

Coverage objective:

$$
H(P)=\sum_{i=1}^{N}\int_{V_i}\|q-p_i\|^2\phi(q)dq
$$

where `V_i` is the Voronoi cell of agent `i`.
Lloyd update moves each agent to its cell centroid.

## Practical Notes

- The integration grid is the algorithm. `CoverageController` takes bounds
  as `[[x_min, y_min], [x_max, y_max]]`; transposing those corners builds
  an **empty** grid, so every Voronoi cell is empty, every centroid equals
  its own agent, and every force is exactly zero. The agents sit still and
  the cost plot is flat — a silent no-op that looks like convergence. The
  constructor now rejects the transposed form rather than accepting it.
- Measure convergence with the **locational cost** `H(P)`, the quantity
  Lloyd actually descends, not the norm of the residual force. Residual
  force says how far the last step moved; it says nothing about whether the
  configuration is any good.

- Convergence depends on bounded domain and update damping.
- Density `\phi(q)` can bias coverage toward high-priority regions.
- Voronoi recomputation cost grows with agent count.

## Implementation and Execution

```bash
python -m uav_sim.simulations.swarm.voronoi_coverage
```

## Evidence

![Voronoi Coverage](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/swarm/voronoi_coverage/voronoi_coverage.gif)

## References

- [Cortes et al., Coverage Control for Mobile Sensing Networks (2004)](https://doi.org/10.1109/TRA.2004.824698)
- [Bullo et al., Distributed Control of Robotic Networks](https://press.princeton.edu/books/hardcover/9780691141954/distributed-control-of-robotic-networks)

## Related Algorithms

- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Potential Swarm](/simulations/swarm/potential-swarm)
