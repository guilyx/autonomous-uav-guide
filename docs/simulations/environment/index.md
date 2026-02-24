<!-- Erwin Lejeune — 2026-02-24 -->
# Environment

Environment algorithms model traversability, obstacle inflation, and online map updates used by planning and control.

## Core Questions

- How should occupancy uncertainty propagate into motion cost?
- What inflation policy best balances safety and corridor width?
- How can dynamic obstacles be fused without planner instability?

## Algorithms

- [Costmap Navigation](/simulations/environment/costmap-navigation)

## Prerequisites

- Occupancy grid mapping
- Distance transforms and inflation layers
- Motion cost design for autonomous navigation
