<!-- Erwin Lejeune — 2026-08-27 -->
# Convoy Escort

## Problem Statement

A ground vehicle drives a fixed route away from a base station. It carries no
long-range radio, so it stays reachable only while a chain of UAVs bridges
the gap — and the gap grows, so the chain has to lengthen, then shorten again
as the route curves back.

## Model and Formulation

Nothing plans the chain. Each aircraft is pulled toward the midpoint of the
link it is responsible for and pushed by the shared connectivity gradient;
the number of hops the route needs is an *outcome*, not an input.

## Tuning and Failure Modes

- **The obvious alternative fails badly.** Keeping the escorts in formation
  around the convoy is what most people reach for first, and it loses the
  base entirely once the convoy is far enough out.
- Hop count is a coarse proxy for exposure: it ignores link margin, so one
  weak hop looks safer than two strong ones.

## Evidence

| escort behaviour | convoy reachable |
|---|---|
| relay chain | **82.1 %** of the run |
| fixed formation | 9.2 % |

Maximum 7 hops at a maximum convoy range of 327 m.

![Convoy Escort](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/comms/convoy_escort/convoy_escort.gif)

## References

- [Scherer and Rinner, Long-term area coverage and radio relay positioning (2018)](https://arxiv.org/abs/1810.12383)
- [CARA: Connectivity-Aware Relay Algorithm (2022)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9737801/)

## Related Algorithms

- [Relay Coverage](/simulations/comms/relay-coverage)
- [Resilient Mesh](/simulations/comms/resilient-mesh)
