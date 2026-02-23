# Erwin Lejeune - 2026-02-21
"""Layered costmap compositor.

Stacks an occupancy grid with inflation, social, and custom layers,
then exposes a unified cost query interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from uav_sim.costmap.inflation_layer import InflationLayer
from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.costmap.social_layer import SocialLayer

if TYPE_CHECKING:
    from uav_sim.environment.obstacles import BoxObstacle
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

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_obstacles(
        cls,
        obstacles: list[BoxObstacle],
        *,
        world_size: float = 30.0,
        resolution: float = 0.5,
        inflation_radius: float = 2.0,
        inflation_method: str = "linear",
    ) -> LayeredCostmap:
        """Build an inflated costmap directly from a list of box obstacles."""
        grid = OccupancyGrid(
            resolution=resolution,
            bounds_min=np.zeros(3),
            bounds_max=np.array([world_size, world_size, 0.0]),
            dimensions=2,
        )
        grid.from_obstacles(obstacles)
        inflation = InflationLayer(
            inflation_radius=inflation_radius,
            method=inflation_method,  # type: ignore[arg-type]
        )
        cm = cls(grid, inflation=inflation)
        cm.update()
        return cm

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

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
