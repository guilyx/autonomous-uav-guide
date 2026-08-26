# Erwin Lejeune - 2026-02-16
"""Voronoi-based area coverage using Lloyd's algorithm.

Reference: J. Cortes et al., "Coverage Control for Mobile Sensing Networks,"
IEEE T-RA, 2004. DOI: 10.1109/TRA.2004.824698
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CoverageController:
    """Lloyd's algorithm for distributed area coverage.

    Iteratively moves agents towards the centroid of their Voronoi
    cell to minimise the coverage cost function.

    Parameters:
        bounds: ``[[x_min, y_min], [x_max, y_max]]`` workspace bounds.
        resolution: Grid resolution for centroid computation.
        gain: Movement gain (0, 1].
    """

    def __init__(
        self,
        bounds: NDArray[np.floating],
        resolution: float = 0.5,
        gain: float = 0.5,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.shape != (2, 2):
            raise ValueError("bounds must be [[x_min, y_min], [x_max, y_max]]")
        if np.any(self.bounds[1] <= self.bounds[0]):
            # Transposing the corners produces an empty integration grid,
            # which makes every centroid equal its own agent and every
            # coverage force exactly zero — a silent no-op rather than a
            # crash. Refuse it instead.
            raise ValueError(
                "bounds must be [[x_min, y_min], [x_max, y_max]] with max > min; "
                f"got {self.bounds.tolist()}"
            )
        self.resolution = resolution
        self.gain = gain

        # Precompute grid points for centroid integration.
        x = np.arange(self.bounds[0, 0], self.bounds[1, 0], resolution)
        y = np.arange(self.bounds[0, 1], self.bounds[1, 1], resolution)
        xx, yy = np.meshgrid(x, y)
        self.grid = np.column_stack([xx.ravel(), yy.ravel()])

    @property
    def region_center(self) -> NDArray[np.floating]:
        """Centre of the coverage region."""
        return self.bounds.mean(axis=0)

    def recenter(self, center: NDArray[np.floating]) -> None:
        """Move the coverage region so it is centred on *center*.

        The region keeps its size and shape; only its position changes.
        That turns static coverage into coverage of a travelling area --
        a patch of ground that moves, with the team redistributing itself
        across it as it goes.

        The integration grid is translated rather than rebuilt. Re-meshing
        it every step would dominate the simulation's runtime for a result
        identical to a vector add, and it would also silently change the
        grid point count whenever the new bounds did not land on the same
        multiple of ``resolution``, which makes the coverage cost jump for
        reasons that have nothing to do with where the agents are.

        Args:
            center: ``[x, y]`` new region centre. Extra components are
                ignored, so a 3D guide point can be passed straight in.
        """
        c = np.asarray(center, dtype=np.float64).reshape(-1)[:2]
        if c.shape != (2,):
            raise ValueError(f"center must have at least 2 components, got {center!r}")
        delta = c - self.region_center
        self.bounds = self.bounds + delta
        self.grid = self.grid + delta

    def compute_centroids(
        self,
        positions_2d: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute Voronoi centroids for each agent (2D).

        Args:
            positions_2d: (N, 2) agent positions in 2D.

        Returns:
            (N, 2) centroid positions.
        """
        N = len(positions_2d)
        centroids = positions_2d.copy()

        if N < 2:
            return centroids

        # Assign each grid point to the nearest agent.
        dists = np.linalg.norm(self.grid[:, None, :] - positions_2d[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)

        for i in range(N):
            mask = assignments == i
            if np.any(mask):
                centroids[i] = np.mean(self.grid[mask], axis=0)

        return centroids

    def empty_cells(self, positions_2d: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Mask of agents whose Voronoi cell contains no grid points.

        An agent far enough outside the region owns none of it, and
        Lloyd's algorithm then has no opinion about where it should go.
        """
        n = len(positions_2d)
        if n < 2 or len(self.grid) == 0:
            return np.zeros(n, dtype=bool)
        dists = np.linalg.norm(self.grid[:, None, :] - positions_2d[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        return ~np.isin(np.arange(n), assignments)

    def compute_forces(
        self,
        positions_2d: NDArray[np.floating],
        recall_outside: bool = True,
    ) -> NDArray[np.floating]:
        """Compute coverage forces (direction towards Voronoi centroid).

        Args:
            positions_2d: (N, 2) current 2D positions.
            recall_outside: Steer agents with an empty Voronoi cell back
                towards the region. Lloyd's algorithm gives such an agent
                a centroid equal to its own position, so its force is
                exactly zero and it is stranded permanently. That never
                arises while the region covers the whole workspace, but
                with a travelling region agents fall outside constantly,
                and without this they are simply abandoned. Pass ``False``
                for textbook-pure Lloyd.

        Returns:
            (N, 2) force vectors.
        """
        centroids = self.compute_centroids(positions_2d)
        forces = self.gain * (centroids - positions_2d)
        if recall_outside:
            stranded = self.empty_cells(positions_2d)
            if np.any(stranded):
                forces[stranded] = self.gain * (self.region_center - positions_2d[stranded])
        return forces

    def coverage_cost(
        self,
        positions_2d: NDArray[np.floating],
    ) -> float:
        """Locational optimisation cost ``∫ min_i ‖q - p_i‖² dq``.

        This is the quantity Lloyd's algorithm descends, so it is the
        honest way to show convergence — unlike the norm of the residual
        force, which says nothing about how good the configuration is.
        """
        if len(positions_2d) == 0 or len(self.grid) == 0:
            return 0.0
        d2 = np.sum((self.grid[:, None, :] - positions_2d[None, :, :]) ** 2, axis=2)
        cell_area = self.resolution**2
        return float(np.sum(np.min(d2, axis=1)) * cell_area)
