# Erwin Lejeune - 2026-02-18
"""Convenience helper: plan an obstacle-aware 3D path through box obstacles.

Builds an occupancy grid from a list of obstacles, inflates it for
safety margin, plans with A*, and returns a smoothed path.  Intended
as a one-call utility for simulations that need an obstacle-free path
without duplicating boilerplate A* setup code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.path_planning.astar_3d import AStar3D
from uav_sim.path_tracking.path_smoothing import smooth_path_3d


def _build_occupancy(
    buildings: list[BoxObstacle], size: int, inflate: int = 1
) -> NDArray[np.bool_]:
    """Convert box obstacles to a 3D boolean grid with optional inflation."""
    grid = np.zeros((size, size, size), dtype=bool)
    for b in buildings:
        lo = np.clip(np.floor(b.min_corner).astype(int) - inflate, 0, size - 1)
        hi = np.clip(np.ceil(b.max_corner).astype(int) + inflate, 0, size)
        grid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return grid


def plan_through_obstacles(
    buildings: list[BoxObstacle],
    start: NDArray[np.floating],
    goal: NDArray[np.floating],
    world_size: int = 30,
    inflate: int = 1,
    smooth_epsilon: float = 2.0,
    smooth_spacing: float = 1.5,
) -> NDArray[np.floating] | None:
    """Plan an A*-smoothed path avoiding box obstacles.

    Parameters
    ----------
    buildings : list of BoxObstacle to avoid.
    start, goal : (3,) world-frame positions.
    world_size : voxel grid side length.
    inflate : inflation radius in voxels for safety margin.
    smooth_epsilon : RDP simplification tolerance.
    smooth_spacing : minimum cubic-spline resampling spacing.

    Returns
    -------
    (N, 3) smoothed waypoint path, or None if no path exists.
    """
    grid = _build_occupancy(buildings, world_size, inflate)
    si = tuple(np.clip(np.round(start).astype(int), 0, world_size - 1))
    gi = tuple(np.clip(np.round(goal).astype(int), 0, world_size - 1))

    # Clear start/goal cells in case inflation clobbered them
    grid[si] = False
    grid[gi] = False

    planner = AStar3D(grid, resolution=1.0)
    path_idx = planner.plan(si, gi)
    if path_idx is None:
        return None

    raw = np.array(path_idx, dtype=float)
    return smooth_path_3d(raw, epsilon=smooth_epsilon, min_spacing=smooth_spacing)
