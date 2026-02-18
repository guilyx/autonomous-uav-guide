# Erwin Lejeune - 2026-02-16
"""3D RRT and RRT* sampling-based path planners.

References:
  [1] S. M. LaValle, "Rapidly-Exploring Random Trees," TR 98-11, 1998.
  [2] S. Karaman, E. Frazzoli, "Sampling-based Algorithms for Optimal
      Motion Planning," IJRR, 2011. DOI: 10.1177/0278364911406761
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RRT3D:
    """Rapidly-exploring Random Tree in 3D space.

    Parameters:
        bounds_min: ``[x_min, y_min, z_min]`` workspace lower bound.
        bounds_max: ``[x_max, y_max, z_max]`` workspace upper bound.
        obstacles: List of (centre, radius) sphere obstacles.
        step_size: Maximum extension distance per iteration.
        goal_radius: Distance threshold to declare goal reached.
        max_iter: Maximum number of iterations.
        goal_bias: Probability of sampling the goal directly.
    """

    def __init__(
        self,
        bounds_min: NDArray[np.floating],
        bounds_max: NDArray[np.floating],
        obstacles: list[tuple[NDArray[np.floating], float]] | None = None,
        step_size: float = 0.5,
        goal_radius: float = 0.3,
        max_iter: int = 5000,
        goal_bias: float = 0.1,
    ) -> None:
        self.bounds_min = np.asarray(bounds_min, dtype=np.float64)
        self.bounds_max = np.asarray(bounds_max, dtype=np.float64)
        self.obstacles = obstacles or []
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iter = max_iter
        self.goal_bias = goal_bias

        self.nodes: list[NDArray[np.floating]] = []
        self.parents: list[int] = []
        self.costs: list[float] = []

    def plan(
        self,
        start: NDArray[np.floating],
        goal: NDArray[np.floating],
        seed: int | None = None,
    ) -> list[NDArray[np.floating]] | None:
        """Find a collision-free path from start to goal.

        Returns:
            List of waypoints from start to goal, or None.
        """
        rng = np.random.default_rng(seed)
        self.nodes = [np.asarray(start, dtype=np.float64)]
        self.parents = [-1]
        self.costs = [0.0]

        for _ in range(self.max_iter):
            q_rand = self._sample(goal, rng)
            idx_near = self._nearest(q_rand)
            q_new = self._steer(self.nodes[idx_near], q_rand)

            if not self._collision_free(self.nodes[idx_near], q_new):
                continue

            cost = self.costs[idx_near] + np.linalg.norm(q_new - self.nodes[idx_near])
            self.nodes.append(q_new)
            self.parents.append(idx_near)
            self.costs.append(float(cost))

            if np.linalg.norm(q_new - goal) < self.goal_radius:
                return self._extract_path(len(self.nodes) - 1)

        return None

    def _sample(
        self, goal: NDArray[np.floating], rng: np.random.Generator
    ) -> NDArray[np.floating]:
        if rng.random() < self.goal_bias:
            return goal.copy()
        return rng.uniform(self.bounds_min, self.bounds_max)

    def _nearest(self, q: NDArray[np.floating]) -> int:
        dists = [np.linalg.norm(n - q) for n in self.nodes]
        return int(np.argmin(dists))

    def _steer(
        self, q_from: NDArray[np.floating], q_to: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        direction = q_to - q_from
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_to.copy()
        return q_from + direction / dist * self.step_size

    def _collision_free(self, q_from: NDArray[np.floating], q_to: NDArray[np.floating]) -> bool:
        """Check if line segment is free of sphere obstacles."""
        for centre, radius in self.obstacles:
            centre = np.asarray(centre, dtype=np.float64)
            d = q_to - q_from
            f = q_from - centre
            a = d @ d
            b = 2 * f @ d
            c = f @ f - radius**2
            disc = b**2 - 4 * a * c
            if disc < 0:
                continue
            disc_sqrt = np.sqrt(disc)
            t1 = (-b - disc_sqrt) / (2 * a + 1e-12)
            t2 = (-b + disc_sqrt) / (2 * a + 1e-12)
            if t1 <= 1.0 and t2 >= 0.0:
                return False
        return True

    def _extract_path(self, idx: int) -> list[NDArray[np.floating]]:
        path = []
        while idx != -1:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        return list(reversed(path))


class RRTStar3D(RRT3D):
    """RRT* with rewiring for asymptotic optimality.

    Parameters:
        gamma: Radius scaling constant. Near radius = γ * (log(n)/n)^(1/3).
    """

    def __init__(self, *args, gamma: float = 5.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    def plan(
        self,
        start: NDArray[np.floating],
        goal: NDArray[np.floating],
        seed: int | None = None,
    ) -> list[NDArray[np.floating]] | None:
        rng = np.random.default_rng(seed)
        self.nodes = [np.asarray(start, dtype=np.float64)]
        self.parents = [-1]
        self.costs = [0.0]
        best_goal_idx: int | None = None

        for _ in range(self.max_iter):
            q_rand = self._sample(goal, rng)
            idx_near = self._nearest(q_rand)
            q_new = self._steer(self.nodes[idx_near], q_rand)

            if not self._collision_free(self.nodes[idx_near], q_new):
                continue

            near_idxs = self._near(q_new)

            # Choose best parent.
            best_parent = idx_near
            best_cost = self.costs[idx_near] + np.linalg.norm(q_new - self.nodes[idx_near])
            for ni in near_idxs:
                new_cost = self.costs[ni] + np.linalg.norm(q_new - self.nodes[ni])
                if new_cost < best_cost and self._collision_free(self.nodes[ni], q_new):
                    best_parent = ni
                    best_cost = new_cost

            new_idx = len(self.nodes)
            self.nodes.append(q_new)
            self.parents.append(best_parent)
            self.costs.append(float(best_cost))

            # Rewire.
            for ni in near_idxs:
                new_cost = best_cost + np.linalg.norm(self.nodes[ni] - q_new)
                if new_cost < self.costs[ni] and self._collision_free(q_new, self.nodes[ni]):
                    self.parents[ni] = new_idx
                    self.costs[ni] = float(new_cost)

            if np.linalg.norm(q_new - goal) < self.goal_radius and (
                best_goal_idx is None or self.costs[new_idx] < self.costs[best_goal_idx]
            ):
                best_goal_idx = new_idx

        if best_goal_idx is not None:
            return self._extract_path(best_goal_idx)
        return None

    def _near(self, q: NDArray[np.floating]) -> list[int]:
        n = len(self.nodes)
        if n == 0:
            return []
        radius = self.gamma * (np.log(n + 1) / (n + 1)) ** (1.0 / 3.0)
        radius = max(radius, self.step_size)
        return [i for i, node in enumerate(self.nodes) if np.linalg.norm(node - q) < radius]
