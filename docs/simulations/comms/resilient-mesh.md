<!-- Erwin Lejeune — 2026-08-27 -->
# Resilient Mesh

## Problem Statement

A connected network is not a survivable one. A chain has healthy algebraic
connectivity right up until any single agent in it fails, at which point it
is two networks. At the halfway mark the agent whose removal costs the most
connectivity — the adversarial choice, not a random one — is switched off.

## Model and Formulation

`degree_of_connectivity` measures survivability directly, by brute-force
removal rather than a spectral proxy: $k = 1$ means one loss splits the mesh,
$k = 2$ means it survives any single loss.

## Tuning and Failure Modes

- **k and λ₂ answer different questions.** k is computed on the *thresholded*
  graph, λ₂ on the *weighted* one. A Gaussian link never reaches exactly
  zero, so λ₂ stays strictly positive even for a fleet with no usable links
  — it merely becomes very small.
- **The threshold matters more than it looks.** At 0.25 a link here needs
  50 m while the median pair sits 70 m apart, which reports a disconnected
  mesh that is in fact doing fine. At 0.10 the same fleet is k = 3.
- **Fail the right agent.** A random failure mostly picks one nobody
  depended on, which proves nothing.

## Evidence

| fleet | worst k before → after | λ₂ after |
|---|---|---|
| tight floor (0.55) | **3 → 2** | 0.688 |
| loose floor (0.24) | **1 → 1** | 0.323 |

Neither fleet splits. The tight fleet keeps k ≥ 2 throughout, so it still
survives *any* single further loss; the loose fleet was a single point of
failure before the loss and remains one. Redundancy is the thing being
measured, and it is not the same question as "is the network up".

k is reported as the **worst value over a trailing six seconds**, matching
the plot. It is an integer that flips as agents drift across the link
threshold, so an instantaneous sample flatters whichever moment it lands on.

![Resilient Mesh](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/comms/resilient_mesh/resilient_mesh.gif)

## References

- [Fiedler, Algebraic connectivity of graphs (1973)](https://dml.cz/handle/10338.dmlcz/101168)
- [Sabattini et al., Decentralized connectivity maintenance (2013)](https://doi.org/10.1177/0278364913499085)

## Related Algorithms

- [Connectivity Maintenance](/simulations/comms/connectivity-maintenance)
- [Convoy Escort](/simulations/comms/convoy-escort)
