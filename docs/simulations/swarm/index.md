<!-- Erwin Lejeune — 2026-02-24 -->
# Swarm

Swarm coordination studies decentralized policies for formation, flocking, consensus, and area coverage.
The chapter emphasizes local rules, communication assumptions, and emergent global behavior.

## One path, six algorithms

Every simulation in this chapter steers the same reference: a figure-8 at
swarm scale, `swarm_figure_8_ref`. They previously each invented their own
motion — two circular orbits, a fixed corner goal, and two that never
translated at all — so comparing them meant first accounting for the fact
that they were not doing the same thing.

A figure-8 rather than an orbit because **a circle lets a formation settle
into one steady bank and hold it forever**, and so never shows whether the
group can be pulled through a reversal and put back together. The crossing
is the part worth watching.

What each algorithm attaches to the guide point differs — a leader, a
virtual structure's origin, a potential well, a coverage region — but the
path they trace does not.

## Core Questions

- What information must be shared for stable formation behavior?
- How do local interaction rules scale with team size?
- Which methods retain robustness under dropouts and disturbances?

## Algorithms

- [Reynolds Flocking](/simulations/swarm/reynolds-flocking)
- [Consensus Formation](/simulations/swarm/consensus-formation)
- [Virtual Structure](/simulations/swarm/virtual-structure)
- [Leader-Follower](/simulations/swarm/leader-follower)
- [Potential Swarm](/simulations/swarm/potential-swarm)
- [Voronoi Coverage](/simulations/swarm/voronoi-coverage)

## Prerequisites

- Graph theory basics for multi-agent systems
- Consensus and stability concepts
- Potential fields and geometric formation models
