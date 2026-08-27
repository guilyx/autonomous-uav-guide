# Erwin Lejeune - 2026-08-27
"""Compose barriers into one safety filter around a nominal controller.

The filter is deliberately a *wrapper*, not a controller. Whatever is
already flying the vehicle keeps flying it, and the filter changes the
command only when a barrier is about to be violated -- by the smallest
amount that keeps it satisfied. That separation is the whole appeal of the
approach: safety can be argued about independently of performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .barriers import Barrier
from .qp import InfeasibleQPError, solve_safety_qp


@dataclass
class FilterReport:
    """What the filter did on one step, for logging and for plots."""

    command: NDArray[np.floating]
    intervened: bool
    correction_norm: float
    margins: dict[str, float] = field(default_factory=dict)
    infeasible: bool = False

    @property
    def worst_margin(self) -> float:
        return min(self.margins.values()) if self.margins else float("inf")


class SafetyFilter:
    """Project a nominal fleet command onto the intersection of the barriers.

    Args:
        barriers: Barriers to enforce. They are stacked into one QP, so a
            solution satisfies all of them simultaneously -- applying them
            in sequence would let the last one undo the others.
        u_min, u_max: Actuator limits, entered as constraints rather than
            applied afterwards. Clipping a filtered command can push it back
            across a barrier the QP had just satisfied.
        fallback: What to command when the QP is infeasible. ``"brake"``
            decelerates along the current velocity, which is the least-bad
            generic response; ``"zero"`` commands nothing; ``"raise"``
            propagates :class:`~flybots.safety.qp.InfeasibleQPError` for
            callers that would rather fail loudly.
    """

    def __init__(
        self,
        barriers: Sequence[Barrier],
        *,
        u_min: float | NDArray[np.floating] | None = None,
        u_max: float | NDArray[np.floating] | None = None,
        fallback: str = "brake",
        brake_gain: float = 4.0,
    ) -> None:
        if fallback not in {"brake", "zero", "raise"}:
            raise ValueError(f"fallback must be brake, zero or raise; got {fallback!r}")
        self.barriers = list(barriers)
        self.u_min = u_min
        self.u_max = u_max
        self.fallback = fallback
        self.brake_gain = float(brake_gain)

    def __call__(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        nominal: NDArray[np.floating],
    ) -> FilterReport:
        """Filter a nominal acceleration command.

        Args:
            positions: ``(N, 3)`` fleet positions.
            velocities: ``(N, 3)`` fleet velocities.
            nominal: ``(N, 3)`` accelerations the nominal controller wants.

        Returns:
            A :class:`FilterReport` carrying the command to apply.
        """
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        u_nom = np.asarray(nominal, dtype=float)
        if p.shape != v.shape or p.shape != u_nom.shape:
            raise ValueError(
                f"positions {p.shape}, velocities {v.shape} and nominal {u_nom.shape} must agree"
            )
        n = len(p)
        flat_nom = u_nom.reshape(-1)

        blocks = [bar.rows(p, v) for bar in self.barriers]
        stacked = [blk for blk in blocks if blk.A.size]
        A = np.vstack([blk.A for blk in stacked]) if stacked else None
        b = np.concatenate([blk.b for blk in stacked]) if stacked else None

        margins = {bar.name: bar.margin(p, v) for bar in self.barriers}

        try:
            flat = solve_safety_qp(flat_nom, A, b, u_min=self.u_min, u_max=self.u_max)
            infeasible = False
        except InfeasibleQPError:
            if self.fallback == "raise":
                raise
            infeasible = True
            flat = (
                np.zeros_like(flat_nom)
                if self.fallback == "zero"
                else (-self.brake_gain * v).reshape(-1)
            )

        correction = float(np.linalg.norm(flat - flat_nom))
        return FilterReport(
            command=flat.reshape(n, 3),
            intervened=correction > 1e-9,
            correction_norm=correction,
            margins=margins,
            infeasible=infeasible,
        )
