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
        goal_saturation: Distance beyond which goal attraction stops
            growing [m].  Unsaturated linear attraction is thousands of
            times stronger than the lattice forces at the start of a long
            transit, so the swarm crosses the map as a disordered blob and
            only forms up once it arrives.  ``0`` disables saturation.
        max_force: Overall clamp on the per-agent force magnitude.
            ``0`` disables it.
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
        goal_saturation: float = 0.0,
        max_force: float = 0.0,
    ) -> None:
        self.d_des = d_des
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.goal_gain = goal_gain
        self.obs_gain = obs_gain
        self.obs_range = obs_range
        self.goal_saturation = goal_saturation
        self.max_force = max_force

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

        # Goal attraction, optionally saturated so it does not swamp the
        # lattice while the swarm is still far from the goal.
        if goal is not None:
            for i in range(N):
                to_goal = goal - positions[i]
                dist = float(np.linalg.norm(to_goal))
                if self.goal_saturation > 0.0 and dist > self.goal_saturation:
                    to_goal = to_goal * (self.goal_saturation / dist)
                forces[i] += self.goal_gain * to_goal

        # Obstacle repulsion.
        if obstacles:
            for i in range(N):
                for centre, radius in obstacles:
                    centre = np.asarray(centre, dtype=np.float64)
                    diff = positions[i] - centre
                    norm = float(np.linalg.norm(diff))
                    if norm < 1e-9:
                        continue
                    # Floor the surface distance: 1/d² is unbounded, and an
                    # agent that clips an obstacle would otherwise get an
                    # infinite kick and leave the world.
                    dist = max(norm - radius, 1e-2)
                    if dist < self.obs_range:
                        forces[i] += self.obs_gain / dist**2 * diff / norm

        if self.max_force > 0.0:
            mags = np.linalg.norm(forces, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.max_force / np.maximum(mags, 1e-12))
            forces *= scale

        return forces
