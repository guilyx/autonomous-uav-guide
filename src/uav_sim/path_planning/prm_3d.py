# Erwin Lejeune - 2026-02-15
"""Probabilistic Roadmap (PRM) 3-D path planner.

Builds a connectivity graph of collision-free samples in configuration
space, then queries shortest paths with Dijkstra.

Reference: L. E. Kavraki et al., "Probabilistic Roadmaps for Path Planning
in High-Dimensional Configuration Spaces," IEEE T-RA, 1996.
DOI: 10.1109/70.508439
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray


class PRM3D:
    """Probabilistic Roadmap planner in 3-D.

    Parameters
    ----------
    bounds_min, bounds_max : workspace extent.
    obstacles : list of ``(centre, radius)`` sphere obstacles.
    n_samples : number of random samples in the roadmap.
    k_neighbours : maximum edges per node (connect to *k* nearest).
    """

    def __init__(
        self,
        bounds_min: NDArray[np.floating],
        bounds_max: NDArray[np.floating],
        obstacles: list[tuple[NDArray[np.floating], float]] | None = None,
        n_samples: int = 400,
        k_neighbours: int = 10,
    ) -> None:
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.obstacles = obstacles or []
        self.n_samples = n_samples
        self.k_neighbours = k_neighbours

        self.nodes: NDArray[np.floating] = np.empty((0, 3))
        self.edges: list[list[tuple[int, float]]] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, seed: int | None = None) -> None:
        """Build the roadmap by sampling and connecting."""
        rng = np.random.default_rng(seed)
        samples: list[NDArray[np.floating]] = []
        while len(samples) < self.n_samples:
            q = rng.uniform(self.bounds_min, self.bounds_max)
            if self._point_free(q):
                samples.append(q)
        self.nodes = np.array(samples)
        n = len(self.nodes)
        self.edges = [[] for _ in range(n)]

        for i in range(n):
            dists = np.linalg.norm(self.nodes - self.nodes[i], axis=1)
            neighbours = np.argsort(dists)[1 : self.k_neighbours + 1]
            for j in neighbours:
                if self._edge_free(self.nodes[i], self.nodes[j]):
                    d = float(dists[j])
                    if not any(nb == j for nb, _ in self.edges[i]):
                        self.edges[i].append((int(j), d))
                        self.edges[j].append((int(i), d))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def plan(
        self,
        start: NDArray[np.floating],
        goal: NDArray[np.floating],
        seed: int | None = None,
    ) -> list[NDArray[np.floating]] | None:
        """Find a collision-free path from *start* to *goal*.

        If the roadmap has not been built yet, :meth:`build` is called
        automatically.

        Returns
        -------
        List of waypoints from start to goal, or ``None``.
        """
        if len(self.nodes) == 0:
            self.build(seed=seed)

        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        si = self._connect_query(start)
        gi = self._connect_query(goal)
        if si is None or gi is None:
            return None

        path_idx = self._dijkstra(si, gi)
        if path_idx is None:
            return None

        path = [start.copy()]
        for idx in path_idx:
            path.append(self.nodes[idx].copy())
        path.append(goal.copy())
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _point_free(self, q: NDArray[np.floating]) -> bool:
        for c, r in self.obstacles:
            if np.linalg.norm(q - np.asarray(c)) < r:
                return False
        return True

    def _edge_free(self, q1: NDArray[np.floating], q2: NDArray[np.floating]) -> bool:
        d = q2 - q1
        length = np.linalg.norm(d)
        if length < 1e-12:
            return True
        n_checks = max(2, int(length / 0.5))
        for t in np.linspace(0.0, 1.0, n_checks):
            if not self._point_free(q1 + t * d):
                return False
        return True

    def _connect_query(self, q: NDArray[np.floating]) -> int | None:
        """Temporarily connect a query point to the nearest visible node."""
        dists = np.linalg.norm(self.nodes - q, axis=1)
        order = np.argsort(dists)
        for idx in order[: self.k_neighbours]:
            if self._edge_free(q, self.nodes[idx]):
                return int(idx)
        return None

    def _dijkstra(self, start_idx: int, goal_idx: int) -> list[int] | None:
        n = len(self.nodes)
        dist = np.full(n, np.inf)
        dist[start_idx] = 0.0
        prev: dict[int, int | None] = {start_idx: None}
        heap: list[tuple[float, int]] = [(0.0, start_idx)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == goal_idx:
                path: list[int] = []
                c: int | None = u
                while c is not None:
                    path.append(c)
                    c = prev.get(c)
                return list(reversed(path))
            if d > dist[u]:
                continue
            for v, w in self.edges[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return None
