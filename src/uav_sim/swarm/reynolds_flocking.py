# Erwin Lejeune - 2026-02-16
"""Reynolds flocking: separation + alignment + cohesion.

Reference: C. W. Reynolds, "Flocks, Herds and Schools: A Distributed
Behavioral Model," SIGGRAPH '87, 1987. DOI: 10.1145/37402.37406
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ReynoldsFlocking:
    """Reynolds boid-style flocking for N agents in 3D.

    Each agent computes a steering force from three rules applied
    to its neighbours within a perception radius.

    Parameters:
        r_percept: Perception radius defining the neighbourhood.
        r_sep: Separation radius (steer away when closer).
        w_sep: Separation weight.
        w_ali: Alignment weight.
        w_coh: Cohesion weight.
    """

    def __init__(
        self,
        r_percept: float = 5.0,
        r_sep: float = 1.5,
        w_sep: float = 2.0,
        w_ali: float = 1.0,
        w_coh: float = 1.0,
    ) -> None:
        self.r_percept = r_percept
        self.r_sep = r_sep
        self.w_sep = w_sep
        self.w_ali = w_ali
        self.w_coh = w_coh

    def compute_forces(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute flocking forces for all agents.

        Args:
            positions: (N, 3) agent positions.
            velocities: (N, 3) agent velocities.

        Returns:
            (N, 3) force vectors for each agent.
        """
        N = len(positions)
        forces = np.zeros_like(positions)

        for i in range(N):
            neighbours = []
            for j in range(N):
                if i == j:
                    continue
                dist = np.linalg.norm(positions[j] - positions[i])
                if dist < self.r_percept:
                    neighbours.append(j)

            if not neighbours:
                continue

            f_sep = np.zeros(3)
            f_ali = np.zeros(3)
            f_coh = np.zeros(3)

            for j in neighbours:
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < self.r_sep and dist > 1e-6:
                    f_sep += diff / dist**2

            avg_vel = np.mean(velocities[neighbours], axis=0)
            f_ali = avg_vel - velocities[i]

            centroid = np.mean(positions[neighbours], axis=0)
            f_coh = centroid - positions[i]

            forces[i] = self.w_sep * f_sep + self.w_ali * f_ali + self.w_coh * f_coh

        return forces
