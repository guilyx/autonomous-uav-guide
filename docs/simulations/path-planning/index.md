<!-- Erwin Lejeune — 2026-02-24 -->
# Path Planning

Path planning produces collision-free, kinematically meaningful routes from start to goal in structured and unstructured 3D environments.

## Core Questions

- How does graph optimality compare against randomized exploration?
- Which planner scales best with map size and obstacle density?
- How should safety margins and smoothness constraints be injected?

## Algorithms

- [A* 3D](/simulations/path-planning/astar-3d)
- [RRT* 3D](/simulations/path-planning/rrt-star-3d)
- [PRM 3D](/simulations/path-planning/prm-3d)
- [Potential Field 3D](/simulations/path-planning/potential-field-3d)
- [Coverage Planning](/simulations/path-planning/coverage-planning)

## Prerequisites

- Graph search and heuristic design
- Sampling-based planning intuition
- Occupancy/costmap representations
