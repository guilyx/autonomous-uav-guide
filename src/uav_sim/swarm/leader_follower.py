# Erwin Lejeune - 2026-02-16
"""Leader-follower formation control.

Reference: J. Desai, J. P. Ostrowski, V. Kumar, "Modeling and Control of
Formations of Nonholonomic Mobile Robots," IEEE T-RA, 2001.
DOI: 10.1109/70.976023
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LeaderFollower:
    """Leader-follower formation controller.

    One leader follows a trajectory autonomously. Each follower
    maintains a desired offset relative to the leader (or its parent
    in a tree topology) using PD control.

    Parameters:
        offsets: (N_followers, 3) desired offsets from the leader.
        kp: Position gain.
        kd: Velocity gain.
    """

    def __init__(
        self,
        offsets: NDArray[np.floating],
        kp: float = 4.0,
        kd: float = 2.0,
    ) -> None:
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.kp = kp
        self.kd = kd
        self.num_followers = len(offsets)

    def compute_forces(
        self,
        leader_pos: NDArray[np.floating],
        leader_vel: NDArray[np.floating],
        follower_positions: NDArray[np.floating],
        follower_velocities: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute PD forces for followers to track their offsets from the leader.

        Args:
            leader_pos: Leader position ``[x, y, z]``.
            leader_vel: Leader velocity ``[vx, vy, vz]``.
            follower_positions: (N, 3) follower positions.
            follower_velocities: (N, 3) follower velocities.

        Returns:
            (N, 3) control forces for each follower.
        """
        desired = leader_pos + self.offsets
        e_pos = desired - follower_positions
        e_vel = leader_vel - follower_velocities
        return self.kp * e_pos + self.kd * e_vel
