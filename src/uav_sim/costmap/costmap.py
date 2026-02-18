# Erwin Lejeune - 2026-02-17
"""Layered costmap compositor.

Stacks an occupancy grid with inflation, social, and custom layers,
then exposes a unified cost query interface.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.costmap.inflation_layer import InflationLayer
from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.costmap.social_layer import SocialLayer
from uav_sim.environment.world import World


class LayeredCostmap:
    """Compose multiple cost layers on top of an :class:`OccupancyGrid`.

    Parameters
    ----------
    grid:
        Base occupancy grid.
    inflation:
        Optional inflation layer.
    social:
        Optional social layer.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        inflation: InflationLayer | None = None,
        social: SocialLayer | None = None,
    ) -> None:
        self.grid = grid
        self.inflation = inflation
        self.social = social
        self._composite: NDArray[np.floating] = grid.grid.copy()

    def update(self, world: World | None = None) -> NDArray[np.floating]:
        """Recompute the composite costmap from all layers."""
        self._composite = self.grid.grid.astype(np.float32).copy()
        if self.inflation is not None:
            self._composite = np.maximum(self._composite, self.inflation.apply(self.grid))
        if self.social is not None and world is not None:
            self._composite = np.maximum(self._composite, self.social.apply(self.grid, world))
        return self._composite

    @property
    def composite(self) -> NDArray[np.floating]:
        return self._composite

    def cost_at(self, point: NDArray[np.floating]) -> float:
        cell = self.grid.world_to_cell(point)
        return float(self._composite[cell])
