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

- **The region travels.** Lloyd on a static box converges once and then has
  nothing left to do. Here a fixed-size region tracks the swarm figure-8, so
  coverage becomes a continuous problem and the team keeps redistributing
  itself as its ground slides away. `recenter` translates the precomputed
  integration grid rather than rebuilding it: re-meshing every step would
  dominate runtime for a result identical to a vector add, and it would
  change the grid point count whenever the new bounds missed a multiple of
  `resolution`, making the cost jump for reasons unrelated to the agents.
- **An agent outside the region is stranded, not merely idle.** Its Voronoi
  cell contains no grid points, so Lloyd hands back a centroid equal to its
  own position and the force is *exactly zero* — it never returns. This
  never arises while the region covers the whole workspace, because every
  agent always owns some of it; with a travelling region agents fall outside
  constantly. `compute_forces` recalls them, and `recall_outside=False`
  keeps textbook-pure Lloyd available for comparison.
- **Feed the region's velocity forward.** The coverage force only ever
  points at where the region is *now*, so without it the team spends the
  whole run chasing and never settles inside. With it the cost falls
  monotonically and stays down while the region moves.
- Convergence depends on bounded domain and update damping.
- Density `\phi(q)` can bias coverage toward high-priority regions.
- Voronoi recomputation cost grows with agent count.

## Implementation and Execution

```bash
python -m flybots.simulations.swarm.voronoi_coverage
```

## Evidence

![Voronoi Coverage](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/swarm/voronoi_coverage/voronoi_coverage.gif)

## References

- [Cortes et al., Coverage Control for Mobile Sensing Networks (2004)](https://doi.org/10.1109/TRA.2004.824698)
- [Bullo et al., Distributed Control of Robotic Networks](https://press.princeton.edu/books/hardcover/9780691141954/distributed-control-of-robotic-networks)

## Related Algorithms

- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Potential Swarm](/simulations/swarm/potential-swarm)
