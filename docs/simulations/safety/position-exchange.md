<!-- Erwin Lejeune — 2026-08-27 -->
# Position Exchange (CBF safety filter)

## Problem Statement

Eight agents sit on a ring, each commanded straight at the point
diametrically opposite it. Every straight-line path crosses the centre at
the same moment, so the nominal controller — a plain PD that cannot see the
other agents — drives the fleet into a single point.

A control barrier function filter sits between that controller and the
vehicles. It plans nothing. It edits the command only when a pair is about
to close inside the safe distance, and only by as much as it must.

## Model and Formulation

The safe set is $h \ge 0$ with

$$h_{ij} = \lVert p_i - p_j \rVert^2 - d_{\text{safe}}^2$$

The squared form is used rather than $\lVert \Delta p \rVert - d$ because it
is smooth everywhere, including the coincident case that the norm form
cannot differentiate — and the coincident case is exactly what a swap drives
towards.

**Relative degree is the thing to get right.** The input is acceleration, so
$\dot h$ contains no input term at all; conditioning on $\dot h \ge -\alpha h$
constrains nothing and the filter silently becomes a pass-through. The
high-order form conditions the second derivative instead:

$$\ddot h + (k_1 + k_2)\dot h + k_1 k_2 h \ge 0$$

Each barrier contributes rows to one fleet-wide program:

$$u^\star = \arg\min_u \tfrac12 \lVert u - u_{\text{nom}} \rVert^2
\quad \text{s.t.} \quad A u \le b$$

Solved as a least-distance program through its non-negative least-squares
dual, which terminates exactly in finitely many steps. Inside a control loop
that matters: a solver that merely *usually* converges is not a safety
guarantee.

## Tuning and Failure Modes

- **A CBF guarantees safety, not liveness.** Standing still satisfies every
  barrier, so the QP will hold a symmetric fleet still indefinitely. Here
  eight agents pressed evenly against each other deadlock 13.9 m from their
  goals, perfectly safely. A tangential bias, faded in only while the fleet
  is jammed, breaks the symmetry and the swap completes; a *constant* swirl
  instead drags every agent off its goal for the whole run.
- **Solve one program for the whole fleet, not one per agent.** A pairwise
  constraint couples both agents' accelerations, so a single QP splits the
  avoidance effort between them. Per-agent programs make each assume the
  other will not move, and both then dodge the same way.
- **Discretisation eats into the margin.** The condition holds in continuous
  time; a forward-Euler step lets it overshoot, and the runs here settle
  0.6–0.9 % inside the barrier. Size the safe distance with room for that
  rather than trusting the number exactly.
- **Enter actuator limits as constraints, not as a clip.** Clipping a solved
  command can walk it straight back through a barrier the QP had just
  satisfied.
- **Infeasibility is a real outcome.** Ask for a separation larger than the
  communication range and no input satisfies both. The filter reports it
  rather than silently returning something unsafe.

## Implementation and Execution

```bash
python -m flybots.simulations.safety.position_exchange
```

## Evidence

| run | closest pair | worst leg error |
|---|---|---|
| unfiltered | **0.051 m** | 0.000 m |
| filtered | 1.799 m | **13.803 m** |
| filtered + swirl | 1.784 m | 0.000 m |

Safe distance 1.800 m. The unfiltered run reaches every goal and passes
through itself to do it; the filtered run never violates the barrier.

The fleet swaps, swaps back, and repeats, so the crossing — the only part
worth watching — recurs through the whole run instead of finishing in five
seconds and leaving forty of frozen ring. That also turns the deadlock from
an anecdote into a pattern: **without the swirl the outbound leg jams every
single time**, while the return leg always completes, because returning is
the direction the jam does not block. The figure quoted is therefore the
worst leg, not the last sample — the run ends on a return leg, which hides
it.

![Position Exchange](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/safety/position_exchange/position_exchange.gif)

## References

- [Ames et al., Control Barrier Function Based Quadratic Programs for Safety Critical Systems (2017)](https://doi.org/10.1109/TAC.2016.2638961)
- [Xiao and Belta, High Order Control Barrier Functions (2022)](https://doi.org/10.1109/TAC.2021.3105491)
- [Borrmann et al., Control Barrier Certificates for Safe Swarm Behavior (2015)](https://doi.org/10.1016/j.ifacol.2015.11.154)

## Related Algorithms

- [Potential Swarm](/simulations/swarm/potential-swarm)
- [Reynolds Flocking](/simulations/swarm/reynolds-flocking)
- [Potential Field 3D](/simulations/path-planning/potential-field-3d)
