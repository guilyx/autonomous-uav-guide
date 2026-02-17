# Erwin Lejeune - 2026-02-16
"""3D potential field path planner with attractive/repulsive forces.

Reference: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and
Mobile Robots," IJRR, 1986. DOI: 10.1177/027836498600500106
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PotentialField3D:
    """Artificial potential field planner in 3D.

    Combines an attractive quadratic potential towards the goal with
    repulsive potentials from obstacles.

    Parameters:
        zeta: Attractive gain.
        eta: Repulsive gain.
        rho0: Obstacle influence distance [m].
        step_size: Gradient descent step size.
        max_iter: Maximum descent iterations.
        goal_tol: Goal reached tolerance [m].
    """

    def __init__(
        self,
        zeta: float = 1.0,
        eta: float = 100.0,
        rho0: float = 2.0,
        step_size: float = 0.1,
        max_iter: int = 5000,
        goal_tol: float = 0.2,
    ) -> None:
        self.zeta = zeta
        self.eta = eta
        self.rho0 = rho0
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_tol = goal_tol

    def plan(
        self,
        start: NDArray[np.floating],
        goal: NDArray[np.floating],
        obstacles: list[tuple[NDArray[np.floating], float]],
    ) -> list[NDArray[np.floating]]:
        """Compute path via gradient descent on the potential field.

        Args:
            start: Start position ``[x, y, z]``.
            goal: Goal position ``[x, y, z]``.
            obstacles: List of ``(centre, radius)`` sphere obstacles.

        Returns:
            List of waypoints from start to goal (may be incomplete
            if trapped in a local minimum).
        """
        pos = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        path = [pos.copy()]

        for _ in range(self.max_iter):
            if np.linalg.norm(pos - goal) < self.goal_tol:
                break
            force = self._attractive_force(pos, goal) + self._repulsive_force(pos, obstacles)
            norm = np.linalg.norm(force)
            if norm < 1e-8:
                # Local minimum escape: random perturbation.
                force = np.random.default_rng().normal(size=3)
                norm = np.linalg.norm(force)
            pos = pos + self.step_size * force / (norm + 1e-12)
            path.append(pos.copy())

        return path

    def _attractive_force(
        self,
        pos: NDArray[np.floating],
        goal: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        return -self.zeta * (pos - goal)

    def _repulsive_force(
        self,
        pos: NDArray[np.floating],
        obstacles: list[tuple[NDArray[np.floating], float]],
    ) -> NDArray[np.floating]:
        force = np.zeros(3)
        for centre, radius in obstacles:
            centre = np.asarray(centre, dtype=np.float64)
            diff = pos - centre
            dist = np.linalg.norm(diff)
            rho = max(dist - radius, 1e-6)
            if rho < self.rho0:
                grad = diff / (dist + 1e-12)
                force += self.eta * (1.0 / rho - 1.0 / self.rho0) * (1.0 / rho**2) * grad
        return force
