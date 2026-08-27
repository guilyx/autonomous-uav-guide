# Erwin Lejeune - 2026-08-27
"""Link models: how good is the radio between two aircraft.

Most swarm papers use a *disk* model -- connected inside a radius, not
outside -- which is convenient and produces a discontinuous graph. Every
connectivity controller worth having differentiates the graph, so a hard
disk gives a gradient that is zero everywhere and undefined at the rim: the
swarm gets no signal that a link is *about* to break, only that it already
has.

The models here are smooth and strictly decreasing in range, so the weight
carries "this link is getting weak" long before it carries "this link is
gone", which is the information a controller can actually act on.

Reference: A. Goldsmith, "Wireless Communications", Cambridge, 2005,
chapter 2 (path loss and shadowing).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = ["GaussianLink", "LinkModel", "PathLossLink"]


class LinkModel(ABC):
    """A smooth, non-negative link weight that decreases with range."""

    @abstractmethod
    def weight(self, distance: NDArray[np.floating]) -> NDArray[np.floating]:
        """Link weight in ``[0, 1]`` for each distance."""

    @abstractmethod
    def dweight_ddistance(self, distance: NDArray[np.floating]) -> NDArray[np.floating]:
        """Derivative of :meth:`weight` with respect to distance."""

    def weights(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Full ``(N, N)`` weighted adjacency, zero on the diagonal."""
        p = np.asarray(positions, dtype=float)
        diff = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        w = self.weight(dist)
        np.fill_diagonal(w, 0.0)
        return w


class GaussianLink(LinkModel):
    """``w = exp(-d² / 2σ²)``.

    The standard choice in the connectivity-maintenance literature: smooth,
    everywhere differentiable, and with a gradient that grows as a link
    stretches. *sigma* is the range at which a link is worth about 0.61.
    """

    def __init__(self, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.sigma = float(sigma)

    def weight(self, distance):
        d = np.asarray(distance, dtype=float)
        return np.exp(-(d**2) / (2.0 * self.sigma**2))

    def dweight_ddistance(self, distance):
        d = np.asarray(distance, dtype=float)
        return -d / self.sigma**2 * self.weight(d)


class PathLossLink(LinkModel):
    """A radio link: received power against a sensitivity floor.

    Free-space-ish path loss ``P_rx ∝ d^-n`` mapped through a logistic on the
    dB margin, so the weight is the probability the link closes rather than a
    geometric convenience. Worth having alongside :class:`GaussianLink`
    because the exponent changes the shape of the swarm: ``n = 2`` in free
    space keeps links usable much further out than ``n = 4`` over ground,
    and a controller tuned on one will spread the fleet wrongly on the other.

    Args:
        reference_range: Range at which the margin is exactly zero, so the
            link is 50/50. Beyond it the weight falls away.
        exponent: Path-loss exponent ``n``. 2 is free space, 3-4 is typical
            over terrain or through clutter.
        softness: dB of margin over which the weight goes from firmly closed
            to firmly open. Small values approach the disk model, with the
            vanishing gradient that implies.
    """

    def __init__(
        self,
        reference_range: float,
        exponent: float = 2.5,
        softness: float = 6.0,
    ) -> None:
        if reference_range <= 0.0:
            raise ValueError(f"reference_range must be positive, got {reference_range}")
        if exponent <= 0.0:
            raise ValueError(f"exponent must be positive, got {exponent}")
        if softness <= 0.0:
            raise ValueError(f"softness must be positive, got {softness}")
        self.reference_range = float(reference_range)
        self.exponent = float(exponent)
        self.softness = float(softness)

    def _margin_db(self, d: NDArray[np.floating]) -> NDArray[np.floating]:
        # Margin relative to the reference range; positive means comfortable.
        safe = np.maximum(d, 1e-9)
        return -10.0 * self.exponent * np.log10(safe / self.reference_range)

    def weight(self, distance):
        d = np.asarray(distance, dtype=float)
        return 1.0 / (1.0 + np.exp(-self._margin_db(d) / self.softness))

    def dweight_ddistance(self, distance):
        d = np.asarray(distance, dtype=float)
        w = self.weight(d)
        safe = np.maximum(d, 1e-9)
        # d(margin)/dd = -10 n / (ln10 · d), and dw/dmargin = w(1-w)/softness.
        dmargin = -10.0 * self.exponent / (np.log(10.0) * safe)
        return w * (1.0 - w) / self.softness * dmargin
