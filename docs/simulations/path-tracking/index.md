<!-- Erwin Lejeune — 2026-02-24 -->
# Path Tracking

Path tracking converts geometric references into feasible attitude and thrust commands.
This chapter spans linear and nonlinear feedback structures used for hover and mission-level behavior.

## Core Questions

- How should positional and attitude loops be decoupled?
- Which error-state definitions improve transient response?
- How do saturation and actuator limits affect stability margins?

## Algorithms

- [PID Hover](/simulations/path-tracking/pid-hover)
- [LQR Hover](/simulations/path-tracking/lqr-hover)
- [Flight Ops Demo](/simulations/path-tracking/flight-ops-demo)

## Prerequisites

- State-space control basics
- Hover linearization around trim
- Attitude representation in SO(3)/Euler form
