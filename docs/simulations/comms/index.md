<!-- Erwin Lejeune — 2026-08-27 -->
# Communications

Most swarm algorithms assume the network. These treat it as part of the
plant: the radio graph has a state, that state degrades as the fleet
spreads, and a controller can act on it before it fails.

## Why a smooth link model matters

The usual *disk* model — connected inside a radius, not outside — is
convenient and useless to a controller. Its gradient is zero everywhere and
undefined at the rim, so the swarm gets no signal that a link is *about* to
break, only that it already has. The models here are smooth and strictly
decreasing in range, so the weight carries "this link is getting weak" long
before it carries "this link is gone".

## Why λ₂ rather than "is it connected"

Connectivity is a boolean, and a boolean tells a controller nothing until it
is already too late. Algebraic connectivity — the second smallest eigenvalue
of the weighted graph Laplacian — is strictly positive exactly while the
network is connected, degrades smoothly as links stretch, and has a closed
form gradient with respect to position. That is what turns connectivity from
a property you check into one you fly.

## Core Questions

- What does the radio actually do at range, and does the model's gradient
  carry usable information before the link fails?
- Which links are load-bearing? The Fiedler vector answers this, and the
  answer is rarely "the longest ones".
- What is coverage worth if it cannot be reported?

## Algorithms

- [Connectivity Maintenance](/simulations/comms/connectivity-maintenance)
- [Relay Coverage](/simulations/comms/relay-coverage)
- [Resilient Mesh](/simulations/comms/resilient-mesh)
- [Link Budget](/simulations/comms/link-budget)
- [Convoy Escort](/simulations/comms/convoy-escort)

## Prerequisites

- Spectral graph theory: Laplacians, eigenvalues, the Fiedler vector
- Radio path loss and link budgets
- Gradient-based multi-agent control
