<!-- Erwin Lejeune — 2026-08-27 -->
# Connectivity Maintenance

## Problem Statement

24 agents are each given a goal drawn at random across a 400 m box. Flown
straight, the task tears the radio network into islands. The question is
whether the fleet can be made to treat its own connectivity as something
worth spending goal progress on.

## Model and Formulation

The network is a weighted graph, the weight falling smoothly with range. Its
health is $\lambda_2$, the second smallest eigenvalue of the Laplacian
$L = D - W$, strictly positive exactly while the graph is connected.

The gradient with respect to position has a closed form through the Fiedler
vector $v$ — the eigenvector belonging to $\lambda_2$:

$$\frac{\partial \lambda_2}{\partial p_i} = \sum_j \frac{\partial w_{ij}}{\partial p_i}\,(v_i - v_j)^2$$

Read that for what it says: effort goes where the **Fiedler vector**
disagrees most, not where the distance is greatest. The Fiedler vector is
near-constant within a tightly connected cluster and jumps across the weak
cut between clusters, so the gradient concentrates on the links actually
holding the network together and ignores redundant ones inside a clump.

The controller scales that gradient by a barrier potential in $\lambda_2$
rather than a fixed gain, so it is nearly silent while the mesh is healthy
and grows without bound as $\lambda_2$ approaches its floor.

## Tuning and Failure Modes

- **A fixed connectivity gain distorts the task everywhere.** Tuned for the
  worst case, it drags on the fleet even when the network is comfortable.
  The barrier form only overrides the task when the mesh is actually at
  risk.
- **Saturate the force.** The barrier diverges by construction; without a
  limit it produces commands no aircraft can fly.
- **A perfectly symmetric formation can produce a repeated $\lambda_2$**,
  where the eigenvector — and so the gradient — is not unique. Any
  disturbance breaks it, but it is why a connectivity controller can look
  briefly erratic in a regular lattice.
- **Watch the topology, not just the number.** A high $\lambda_2$ in a chain
  is still a chain: $k$-connectivity of 1 means every agent is a single
  point of failure.

## Implementation and Execution

```bash
python -m flybots.simulations.comms.connectivity_maintenance
```

## Evidence

| run | final $\lambda_2$ | $k$-connectivity |
|---|---|---|
| task only | 5.67e-06 | 0 (fragmented) |
| connectivity-aware | 0.2762 | 1 |

The aware run does **not** reach its goals — mean goal error plateaus near
110 m while the task-only run drives it to zero. Connectivity was bought
with mission progress, and the second panel shows the bill.

![Connectivity Maintenance](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/comms/connectivity_maintenance/connectivity_maintenance.gif)

## References

- [Sabattini et al., Decentralized connectivity maintenance (2013)](https://doi.org/10.1177/0278364913499085)
- [De Gennaro and Jadbabaie, Decentralized Control of Connectivity (2006)](https://doi.org/10.1109/CDC.2006.377041)
- [Optimal Multi-Robot Communication-Aware Trajectory Planning by Constraining the Fiedler Value (2024)](https://arxiv.org/abs/2406.18452)

## Related Algorithms

- [Relay Coverage](/simulations/comms/relay-coverage)
- [Consensus Formation](/simulations/swarm/consensus-formation)
