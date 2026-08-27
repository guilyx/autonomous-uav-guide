<!-- Erwin Lejeune — 2026-08-27 -->
# Connectivity Annulus

## Problem Statement

Two barriers that pull opposite ways. `SafeDistanceBarrier` pushes every pair
apart; `ConnectivityBarrier` pulls every pair together. Run them at once and
the fleet is squeezed into a shell — close enough to talk, far enough not to
touch — without anything having planned that shape. It falls out of the
intersection of two safe sets.

## Model and Formulation

$h_{\text{sep}} = \lVert \Delta p \rVert^2 - d^2$ and
$h_{\text{conn}} = R^2 - \lVert \Delta p \rVert^2$ — the same quantity with
opposite sign. Both have relative degree two, both become rows of one QP, and
a solution satisfies both simultaneously.

## Tuning and Failure Modes

- **Start inside the safe set.** A CBF guarantees forward *invariance*: it
  holds a set you are already in and cannot undo a violated initial
  condition. Seeding at random put three pairs inside the safe distance at
  t = 0, and the run then reported that as its minimum — which reads as a
  barrier failure when it is nothing of the kind.
- **Size the initial ring against both bounds.** On a circle of radius $R$
  with $N$ agents the closest pair is $2R\sin(\pi/N)$ and the furthest is
  $2R$. Sizing off the separation alone put diametric pairs at 28 m against a
  26 m radio: satisfying one barrier by breaking the other.
- **Infeasibility is a real outcome.** Ask for more separation than the range
  allows and no input satisfies both.
- **A static scene proves nothing.** Pushing outward against a fixed range
  reaches equilibrium in about thirteen seconds and then nothing moves.
  Flying the formation along a figure-8 instead put it in motion but pulled
  it together: the guide attraction beat the outward push, the furthest pair
  collapsed to half the range, and the connectivity barrier stopped being
  active at all — the shell, the entire point, quietly disappeared. Varying
  the *range* rather than the position keeps both barriers working.

## Evidence

| quantity | value | limit |
|---|---|---|
| closest pair | **6.00 m** | safe distance 6.0 m |
| furthest pair | **31.94 m** | comm range 32.0 m (peak) |

The radio range **breathes** between 12 m and 32 m and the shell breathes
with it: the fleet expands to fill whatever range it is given and contracts
as that range collapses, never once closing inside the safe distance.
Nothing schedules that — the barrier is simply time-varying and the QP
re-solves against it each step.

The impossible case reports infeasible on **2200/2200** steps rather than
returning something unsafe.

![Connectivity Annulus](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/safety/connectivity_annulus/connectivity_annulus.gif)

## References

- [Ames et al., CBF-based Quadratic Programs (2017)](https://doi.org/10.1109/TAC.2016.2638961)
- [Borrmann et al., Control Barrier Certificates for Safe Swarm Behavior (2015)](https://doi.org/10.1016/j.ifacol.2015.11.154)

## Related Algorithms

- [Position Exchange](/simulations/safety/position-exchange)
- [Connectivity Maintenance](/simulations/comms/connectivity-maintenance)
