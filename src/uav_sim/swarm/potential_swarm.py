# Erwin Lejeune - 2026-02-16
"""Potential-based swarm navigation with Lennard-Jones inter-agent potential.

Reference: W. M. Spears et al., "Distributed, Physics-Based Control of
Swarms of Vehicles," Autonomous Robots, 2004. DOI: 10.1023/B:AURO.0000033971.96584.f2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PotentialSwarm:
    """Lennard-Jones-like potential for swarm navigation.

    Inter-agent potential provides equilibrium at desired spacing.
    Goal attraction and obstacle repulsion are added for navigation.

    Parameters:
        d_des: Desired inter-agent distance [m].
        epsilon: Potential well depth.
        a: Repulsive exponent.
        b: Attractive exponent (a > b).
        goal_gain: Goal attraction gain.
        obs_gain: Obstacle repulsion gain.
        obs_range: Obstacle influence range [m].
    """

    def __init__(
        self,
        d_des: float = 2.0,
        epsilon: float = 5.0,
        a: int = 4,
        b: int = 2,
        goal_gain: float = 1.0,
        obs_gain: float = 50.0,
        obs_range: float = 3.0,
    ) -> None:
        self.d_des = d_des
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.goal_gain = goal_gain
        self.obs_gain = obs_gain
        self.obs_range = obs_range

    def compute_forces(
        self,
        positions: NDArray[np.floating],
        goal: NDArray[np.floating] | None = None,
        obstacles: list[tuple[NDArray[np.floating], float]] | None = None,
    ) -> NDArray[np.floating]:
        """Compute potential-based forces for all agents.

        Args:
            positions: (N, 3) agent positions.
            goal: Optional goal position (shared by all agents).
            obstacles: Optional list of ``(centre, radius)`` spheres.

        Returns:
            (N, 3) force vectors.
        """
        N = len(positions)
        forces = np.zeros_like(positions)

        # Inter-agent potential.
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = positions[i] - positions[j]
                r = np.linalg.norm(diff)
                if r < 1e-6:
                    continue
                # Lennard-Jones force (negative gradient).
                f_mag = (
                    self.epsilon
                    * (
                        self.a * (self.d_des / r) ** (self.a + 1)
                        - self.b * (self.d_des / r) ** (self.b + 1)
                    )
                    / r
                )
                forces[i] += f_mag * diff / r

        # Goal attraction.
        if goal is not None:
            for i in range(N):
                forces[i] += self.goal_gain * (goal - positions[i])

        # Obstacle repulsion.
        if obstacles:
            for i in range(N):
                for centre, radius in obstacles:
                    centre = np.asarray(centre, dtype=np.float64)
                    diff = positions[i] - centre
                    dist = np.linalg.norm(diff) - radius
                    if dist < self.obs_range and dist > 1e-6:
                        forces[i] += self.obs_gain / dist**2 * diff / np.linalg.norm(diff)

        return forces
