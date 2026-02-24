<!-- Erwin Lejeune — 2026-02-24 -->
# Trajectory Planning

Trajectory planning converts geometric paths into smooth, dynamically feasible time-parameterized references.
The focus is continuity, comfort, and actuator-aware feasibility.

## Core Questions

- How much continuity is required at waypoints?
- Which objective balances smoothness versus travel time?
- How should constraints be encoded for aggressive maneuvers?

## Algorithms

- [Minimum Snap](/simulations/trajectory-planning/min-snap)
- [Polynomial Trajectory](/simulations/trajectory-planning/polynomial)
- [Quintic Polynomial](/simulations/trajectory-planning/quintic)
- [Frenet Optimal](/simulations/trajectory-planning/frenet-optimal)

## Prerequisites

- Differential flatness intuition for quadrotors
- Polynomial optimization and boundary conditions
- Curvilinear coordinates (Frenet frames)
