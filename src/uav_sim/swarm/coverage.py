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
        self.resolution = resolution
        self.gain = gain

        # Precompute grid points for centroid integration.
        x = np.arange(self.bounds[0, 0], self.bounds[1, 0], resolution)
        y = np.arange(self.bounds[0, 1], self.bounds[1, 1], resolution)
        xx, yy = np.meshgrid(x, y)
        self.grid = np.column_stack([xx.ravel(), yy.ravel()])

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

    def compute_forces(
        self,
        positions_2d: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute coverage forces (direction towards Voronoi centroid).

        Args:
            positions_2d: (N, 2) current 2D positions.

        Returns:
            (N, 2) force vectors.
        """
        centroids = self.compute_centroids(positions_2d)
        return self.gain * (centroids - positions_2d)
