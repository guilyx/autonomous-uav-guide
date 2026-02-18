# Erwin Lejeune - 2026-02-17
"""Obstacle inflation layer for safe-distance costmaps.

Applies a distance-based cost decay around each occupied cell, ensuring
the planner keeps a minimum clearance from obstacles.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from uav_sim.costmap.occupancy_grid import OccupancyGrid


class InflationLayer:
    """Inflate occupied cells by a distance-decaying cost.

    Parameters
    ----------
    inflation_radius:
        Maximum inflation distance [m].
    cost_scaling:
        Exponential decay factor: ``cost = exp(-cost_scaling * d)``.
    """

    def __init__(self, inflation_radius: float = 2.0, cost_scaling: float = 2.0) -> None:
        self.inflation_radius = inflation_radius
        self.cost_scaling = cost_scaling

    def apply(self, grid: OccupancyGrid) -> NDArray[np.floating]:
        """Return an inflated cost grid (same shape as the occupancy grid)."""
        binary = (grid.grid >= 0.5).astype(np.float32)
        dist = distance_transform_edt(1.0 - binary) * grid.resolution
        inflated = np.where(
            dist <= self.inflation_radius,
            np.exp(-self.cost_scaling * dist),
            0.0,
        ).astype(np.float32)
        return np.maximum(binary, inflated)
