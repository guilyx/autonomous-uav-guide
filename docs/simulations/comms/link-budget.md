<!-- Erwin Lejeune — 2026-08-27 -->
# Link Budget

## Problem Statement

Same fleet, same controller, same gains. Only the path-loss exponent changes:
$n = 2$ for free space, $n = 4$ for propagation over terrain and through
clutter.

## A negative result

This simulation was built to show that a fleet tuned on a free-space
assumption fragments when flown somewhere cluttered. **It does not**, and the
negative result is the more useful one.

| configuration | λ₂ |
|---|---|
| matched clutter (n=4 tuned, n=4 flown) | 1.562 |
| mismatched (n=2 tuned, n=4 flown) | **1.557** |

A difference of a third of a percent. The connectivity gradient is dominated
by *geometry*, not by the exponent: both models are smooth and monotone in
range, so they rank the links in the same order and push in nearly the same
direction. The exponent changes the magnitudes, and the barrier potential on
λ₂ renormalises most of that away.

Getting the link model wrong costs surprisingly little — worth knowing
before spending effort characterising a radio to three decimal places.

## What the exponent does change

| | spread | λ₂ |
|---|---|---|
| free space | 94.7 m | **3.98** |
| clutter | 94.6 m | **1.71** |

Identical formation, less than half the connectivity. A margin computed in
one world does not transfer to the other, even though the control does.

## Evidence

![Link Budget](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/comms/link_budget/link_budget.gif)

## References

- A. Goldsmith, *Wireless Communications*, Cambridge, 2005, ch. 2

## Related Algorithms

- [Connectivity Maintenance](/simulations/comms/connectivity-maintenance)
- [Resilient Mesh](/simulations/comms/resilient-mesh)
