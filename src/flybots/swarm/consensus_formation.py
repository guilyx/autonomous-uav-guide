# Erwin Lejeune - 2026-02-16
"""Consensus-based formation control via graph Laplacian.

Reference: R. Olfati-Saber, R. M. Murray, "Consensus Problems in Networks
of Agents with Switching Topology," IEEE TAC, 2004. DOI: 10.1109/TAC.2004.834113
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ConsensusFormation:
    """Distributed consensus formation using graph Laplacian.

    Each agent converges to a desired formation offset from the
    group centroid using the protocol:
    ``u_i = -Σ_{j ∈ N_i} a_ij * ((p_i - δ_i) - (p_j - δ_j))``

    Parameters:
        adjacency: (N, N) adjacency matrix of the communication graph.
        offsets: (N, 3) desired formation offsets.
        gain: Control gain scalar.
    """

    def __init__(
        self,
        adjacency: NDArray[np.floating],
        offsets: NDArray[np.floating],
        gain: float = 1.0,
    ) -> None:
        self.adjacency = np.asarray(adjacency, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.gain = gain
        self.N = len(adjacency)

        degree = np.diag(np.sum(self.adjacency, axis=1))
        self.laplacian = degree - self.adjacency

    def compute_forces(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute consensus control forces for all agents.

        Args:
            positions: (N, 3) current positions.

        Returns:
            (N, 3) control forces.
        """
        errors = positions - self.offsets
        forces = np.zeros_like(positions)
        for d in range(3):
            forces[:, d] = -self.gain * self.laplacian @ errors[:, d]
        return forces
