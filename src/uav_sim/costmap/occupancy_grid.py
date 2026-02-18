# Erwin Lejeune - 2026-02-17
"""2-D / 3-D occupancy grid with binary or probabilistic cells.

Reference: S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor
Models," Autonomous Robots, 2003. DOI: 10.1023/A:1024972020450
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.world import World


class OccupancyGrid:
    """Voxelised occupancy representation of a :class:`World`.

    Parameters
    ----------
    resolution:
        Edge length of each cell [m].
    bounds_min, bounds_max:
        Workspace extents.
    dimensions:
        2 for a height-collapsed 2-D grid, 3 for a full 3-D voxel grid.
    """

    def __init__(
        self,
        resolution: float = 0.5,
        bounds_min: NDArray[np.floating] | None = None,
        bounds_max: NDArray[np.floating] | None = None,
        dimensions: int = 2,
    ) -> None:
        self.resolution = resolution
        self.bounds_min = np.asarray(bounds_min if bounds_min is not None else np.zeros(3))
        self.bounds_max = np.asarray(bounds_max if bounds_max is not None else np.full(3, 50.0))
        self.dimensions = dimensions

        sizes = ((self.bounds_max - self.bounds_min) / resolution).astype(int) + 1
        if dimensions == 2:
            self._grid = np.zeros((sizes[0], sizes[1]), dtype=np.float32)
        else:
            self._grid = np.zeros((sizes[0], sizes[1], sizes[2]), dtype=np.float32)

    @property
    def grid(self) -> NDArray[np.floating]:
        return self._grid

    @property
    def shape(self) -> tuple[int, ...]:
        return self._grid.shape

    def world_to_cell(self, point: NDArray[np.floating]) -> tuple[int, ...]:
        idx = (
            (point[: self.dimensions] - self.bounds_min[: self.dimensions]) / self.resolution
        ).astype(int)
        return tuple(np.clip(idx, 0, np.array(self._grid.shape) - 1))

    def cell_to_world(self, cell: tuple[int, ...]) -> NDArray[np.floating]:
        return self.bounds_min[: len(cell)] + np.array(cell) * self.resolution + self.resolution / 2

    def set_occupied(self, point: NDArray[np.floating], value: float = 1.0) -> None:
        self._grid[self.world_to_cell(point)] = value

    def is_occupied(self, point: NDArray[np.floating], threshold: float = 0.5) -> bool:
        return bool(self._grid[self.world_to_cell(point)] >= threshold)

    def from_world(self, world: World) -> None:
        """Rasterise all obstacles in *world* into this grid."""
        it = np.nditer(self._grid, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            if self.dimensions == 2:
                pt = self.cell_to_world(idx)
                pt3 = np.array([pt[0], pt[1], 0.0])
            else:
                pt3 = self.cell_to_world(idx)
            if not world.is_free(pt3):
                self._grid[idx] = 1.0
            it.iternext()
