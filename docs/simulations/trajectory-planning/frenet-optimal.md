<!-- Erwin Lejeune — 2026-02-24 -->
# Frenet Optimal Trajectory

## Problem Statement

In corridor-like navigation, curvilinear coordinates simplify trajectory generation relative to a reference path.
Frenet planning separates longitudinal and lateral behavior, enabling efficient candidate generation and selection.

## Model and Formulation

State in Frenet frame:

$$
s: \text{arc length along reference},\quad d: \text{lateral offset}
$$

Candidates are built from polynomial models:

- longitudinal `s(t)` via quartic/quintic forms
- lateral `d(t)` via quintic forms

Total cost aggregates smoothness, offset, speed-tracking, and collision penalties:

$$
J = w_d J_d + w_s J_s + w_c J_{collision}
$$

## Algorithm Procedure

1. Generate candidate lateral and longitudinal profiles.
2. Transform each candidate back to Cartesian coordinates.
3. Evaluate dynamic, collision, and comfort constraints.
4. Select minimum-cost feasible trajectory.

## Tuning Guidance

- Increase collision term when operating near dense obstacles.
- Penalize lateral offset more heavily for narrow corridors.
- Short horizons improve responsiveness but can increase myopic behavior.

## Failure Modes and Diagnostics

- Poor reference-path quality leads to unstable Frenet projections.
- Candidate set too small misses globally better maneuvers.
- Fast curvature changes can violate dynamic feasibility.

## Implementation and Execution

```bash
python -m uav_sim.simulations.trajectory_planning.frenet_optimal
```

## Evidence

![Frenet Optimal](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/trajectory_planning/frenet_optimal/frenet_optimal.gif)

## References

- [Werling et al., Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame (2010)](https://doi.org/10.1109/ROBOT.2010.5509799)
- [Bender et al., The Urban Challenge: Integrating Planning and Control](https://doi.org/10.1109/TITS.2014.2301706)

## Related Algorithms

- [A* 3D](/simulations/path-planning/astar-3d)
- [Quintic Polynomial](/simulations/trajectory-planning/quintic)
