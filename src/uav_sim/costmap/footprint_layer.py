# Erwin Lejeune - 2026-02-18
"""Footprint-aware inflation layer for costmaps.

Uses the vehicle :class:`BaseFootprint` bounding radius to determine
the inflation distance, producing a safety margin that exactly matches
the robot's physical extent.

This is particularly useful for swarm envelopes where a convex hull
footprint around multiple agents defines the inflation boundary.

Inspired by ``simple_autonomous_car``'s footprint-based costmap.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.vehicles.footprint import BaseFootprint


class FootprintInflationLayer:
    """Inflate obstacles by the vehicle footprint bounding radius.

    Parameters
    ----------
    footprint : the robot's physical extent.
    padding : extra clearance beyond the bounding radius [m].
    cost_scaling : exponential decay factor.
    """

    def __init__(
        self,
        footprint: BaseFootprint,
        padding: float = 0.2,
        cost_scaling: float = 2.0,
    ) -> None:
        self.footprint = footprint
        self.padding = padding
        self.cost_scaling = cost_scaling

    @property
    def inflation_radius(self) -> float:
        return self.footprint.inflation_radius(self.padding)

    def apply(self, grid: OccupancyGrid) -> NDArray[np.floating]:
        """Return an inflated cost grid sized to the footprint."""
        binary = (grid.grid >= 0.5).astype(np.float32)
        dist = distance_transform_edt(1.0 - binary) * grid.resolution
        r = self.inflation_radius
        inflated = np.where(
            dist <= r,
            np.exp(-self.cost_scaling * dist),
            0.0,
        ).astype(np.float32)
        return np.maximum(binary, inflated)
