# Erwin Lejeune - 2026-02-15
"""2-D / 3-D occupancy grid with frame-aware queries and visualisation.

Supports both global and ego-centred costmap representations, following the
same pattern as ``simple_autonomous_car.costmap.GridCostmap``.

Reference: S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor
Models," Autonomous Robots, 2003. DOI: 10.1023/A:1024972020450
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from uav_sim.frames import FrameID, body_to_world, world_to_body

if TYPE_CHECKING:
    from uav_sim.environment.obstacles import BoxObstacle
    from uav_sim.environment.world import World

COSTMAP_CMAP = "RdYlGn_r"
COST_FREE = 0.0
COST_OCCUPIED = 1.0


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
    frame:
        ``FrameID.WORLD`` for a fixed global grid,
        ``FrameID.BODY`` for an ego-centred grid that moves with the vehicle.
    """

    def __init__(
        self,
        resolution: float = 0.5,
        bounds_min: NDArray[np.floating] | None = None,
        bounds_max: NDArray[np.floating] | None = None,
        dimensions: int = 2,
        frame: FrameID = FrameID.WORLD,
    ) -> None:
        self.resolution = resolution
        self.bounds_min = np.asarray(bounds_min if bounds_min is not None else np.zeros(3))
        self.bounds_max = np.asarray(bounds_max if bounds_max is not None else np.full(3, 50.0))
        self.dimensions = dimensions
        self.frame = frame

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
        return (
            self.bounds_min[: len(cell)] + np.array(cell) * self.resolution + self.resolution / 2
        )

    def set_occupied(self, point: NDArray[np.floating], value: float = 1.0) -> None:
        self._grid[self.world_to_cell(point)] = value

    def is_occupied(self, point: NDArray[np.floating], threshold: float = 0.5) -> bool:
        return bool(self._grid[self.world_to_cell(point)] >= threshold)

    def get_cost(
        self,
        point: NDArray[np.floating],
        *,
        query_frame: FrameID = FrameID.WORLD,
        vehicle_position: NDArray[np.floating] | None = None,
        vehicle_euler: NDArray[np.floating] | None = None,
    ) -> float:
        """Return the continuous cost [0, 1] at *point*.

        If the grid is in world frame and the query is in body frame (or vice
        versa), the caller must supply vehicle pose for the conversion.
        """
        pt = self._resolve_frame(point, query_frame, vehicle_position, vehicle_euler)
        cell = self.world_to_cell(pt)
        return float(self._grid[cell])

    def clear(self) -> None:
        """Reset all cells to free space."""
        self._grid.fill(COST_FREE)

    # ------------------------------------------------------------------
    # Rasterisation
    # ------------------------------------------------------------------

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
                self._grid[idx] = COST_OCCUPIED
            it.iternext()

    def from_obstacles(self, obstacles: list[BoxObstacle]) -> None:
        """Rasterise *obstacles* (box list) into this grid — faster than from_world."""
        for b in obstacles:
            lo = np.clip(
                (
                    (b.min_corner[: self.dimensions] - self.bounds_min[: self.dimensions])
                    / self.resolution
                ).astype(int),
                0,
                np.array(self._grid.shape) - 1,
            )
            hi = np.clip(
                (
                    (b.max_corner[: self.dimensions] - self.bounds_min[: self.dimensions])
                    / self.resolution
                ).astype(int)
                + 1,
                0,
                np.array(self._grid.shape),
            )
            if self.dimensions == 2:
                self._grid[lo[0] : hi[0], lo[1] : hi[1]] = COST_OCCUPIED
            else:
                self._grid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = COST_OCCUPIED

    def from_points(
        self,
        points: NDArray[np.floating],
        *,
        query_frame: FrameID = FrameID.WORLD,
        vehicle_position: NDArray[np.floating] | None = None,
        vehicle_euler: NDArray[np.floating] | None = None,
    ) -> None:
        """Mark cells hit by an (N, 2) or (N, 3) point cloud as occupied.

        Useful for building a local costmap from lidar returns.
        """
        for pt in points:
            resolved = self._resolve_frame(pt, query_frame, vehicle_position, vehicle_euler)
            cell = self.world_to_cell(resolved)
            valid = all(0 <= cell[d] < self._grid.shape[d] for d in range(len(cell)))
            if valid:
                self._grid[cell] = COST_OCCUPIED

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def _resolve_frame(
        self,
        point: NDArray[np.floating],
        query_frame: FrameID,
        vehicle_position: NDArray[np.floating] | None,
        vehicle_euler: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        """Convert *point* from *query_frame* into the grid's native frame."""
        if query_frame == self.frame:
            return np.asarray(point)

        if vehicle_position is None or vehicle_euler is None:
            raise ValueError("vehicle_position and vehicle_euler required for cross-frame queries")

        point3 = np.zeros(3)
        point3[: len(point)] = point

        if query_frame == FrameID.BODY and self.frame == FrameID.WORLD:
            return body_to_world(point3, vehicle_position, vehicle_euler)
        if query_frame == FrameID.WORLD and self.frame == FrameID.BODY:
            return world_to_body(point3, vehicle_position, vehicle_euler)

        return np.asarray(point)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def visualize_3d(
        self,
        ax: object,
        *,
        cmap: str = COSTMAP_CMAP,
        alpha: float = 0.35,
        z_height: float = 0.0,
        stride: int = 2,
    ) -> None:
        """Render the 2-D occupancy grid as a coloured floor surface on an Axes3D."""
        import matplotlib.pyplot as plt

        g = self._grid
        if g.ndim != 2:
            g = g.max(axis=2) if g.ndim == 3 else g

        cm = plt.get_cmap(cmap)
        nx, ny = g.shape
        xg = np.linspace(float(self.bounds_min[0]), float(self.bounds_max[0]), nx + 1)
        yg = np.linspace(float(self.bounds_min[1]), float(self.bounds_max[1]), ny + 1)
        xm, ym = np.meshgrid(xg, yg, indexing="ij")
        zz = np.full_like(xm, z_height)

        face_colors = cm(g[..., np.newaxis].repeat(1, axis=2))[:, :, 0]
        ax.plot_surface(  # type: ignore[union-attr]
            xm,
            ym,
            zz,
            facecolors=face_colors,
            alpha=alpha,
            rstride=stride,
            cstride=stride,
            linewidth=0,
            antialiased=False,
        )

    def visualize_2d(
        self,
        ax: object,
        *,
        cmap: str = COSTMAP_CMAP,
        vmin: float = 0.0,
        vmax: float = 1.0,
        alpha: float = 0.45,
        mask_free: bool = True,
    ) -> object:
        """Render the 2-D grid as a translucent costmap overlay.

        Green = free, yellow = inflated, red = occupied.

        When *mask_free* is True, zero-cost cells are made transparent
        (matching ``simple_autonomous_car``'s visualisation style).

        Returns the ``AxesImage`` so callers can call ``set_data`` for animation.
        """
        g = self._grid
        if g.ndim != 2:
            g = g.max(axis=2) if g.ndim == 3 else g

        display = (
            np.ma.masked_where(g <= COST_FREE, g) if (mask_free and np.any(g > COST_FREE)) else g
        )

        extent = [
            float(self.bounds_min[0]),
            float(self.bounds_max[0]),
            float(self.bounds_min[1]),
            float(self.bounds_max[1]),
        ]
        return ax.imshow(  # type: ignore[union-attr]
            display.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            zorder=0,
        )
