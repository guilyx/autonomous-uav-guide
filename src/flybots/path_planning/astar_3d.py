# Erwin Lejeune - 2026-02-16
"""3D A* grid-based path planner.

Reference: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the
Heuristic Determination of Minimum Cost Paths," IEEE TSSC, 1968.
DOI: 10.1109/TSSC.1968.300136
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray


class AStar3D:
    """A* search on a 3D voxel grid.

    Parameters:
        grid: 3D boolean array where ``True`` = obstacle.
        resolution: Grid cell size [m].
    """

    # 26-connected neighbourhood offsets.
    _NEIGHBOURS = np.array(
        [
            [dx, dy, dz]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    )

    def __init__(self, grid: NDArray[np.bool_], resolution: float = 1.0) -> None:
        self.grid = grid
        self.resolution = resolution

    def plan(
        self,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
    ) -> list[tuple[int, int, int]] | None:
        """Find shortest obstacle-free path from start to goal.

        Args:
            start: (i, j, k) grid index.
            goal: (i, j, k) grid index.

        Returns:
            List of grid indices from start to goal, or None if no path.
        """
        if self.grid[start] or self.grid[goal]:
            return None

        open_set: list[tuple[float, tuple[int, int, int]]] = []
        heapq.heappush(open_set, (0.0, start))
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], float] = {start: 0.0}

        shape = self.grid.shape

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct(came_from, current)

            for d in self._NEIGHBOURS:
                nb = (current[0] + d[0], current[1] + d[1], current[2] + d[2])

                if not (0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and 0 <= nb[2] < shape[2]):
                    continue
                if self.grid[nb]:
                    continue

                cost = np.linalg.norm(d) * self.resolution
                tentative_g = g_score[current] + cost

                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, goal)
                    heapq.heappush(open_set, (f, nb))

        return None

    def _heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        """Euclidean distance heuristic."""
        return (
            float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))
            * self.resolution
        )

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int, int], tuple[int, int, int]],
        current: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
