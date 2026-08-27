# Erwin Lejeune - 2026-08-27
"""Control barrier functions for a fleet of double-integrator vehicles.

A barrier is a scalar ``h`` over the state, defined so that ``h ≥ 0`` is the
safe set. Each barrier here turns itself into linear inequalities on the
stacked acceleration command, which the QP in :mod:`flybots.safety.qp` then
enforces while staying as close as possible to the nominal controller.

The state is the fleet's positions ``(N, 3)`` and velocities ``(N, 3)``; the
decision variable is the stacked accelerations, flattened to ``(3N,)``.

**Relative degree is the thing to get right.** Acceleration is the input, so
a barrier on *position* -- keep two vehicles apart, stay inside a fence --
has no acceleration term in its first derivative: ``ḣ`` depends only on
velocity, and the input cannot appear until ``ḧ``. Enforcing the plain
condition ``ḣ ≥ −α(h)`` on such a barrier constrains nothing at all, and the
filter silently becomes a pass-through. Those barriers use the high-order
form (Xiao & Belta) instead. Barriers on *velocity*, like a speed limit, do
have relative degree one and use the plain form.

References:
- A. D. Ames et al., "Control Barrier Function Based Quadratic Programs for
  Safety Critical Systems," IEEE TAC, 2017. DOI: 10.1109/TAC.2016.2638961
- W. Xiao & C. Belta, "High Order Control Barrier Functions," IEEE TAC,
  2022. DOI: 10.1109/TAC.2021.3105491
- U. Borrmann et al., "Control Barrier Certificates for Safe Swarm
  Behavior," IFAC ADHS, 2015. DOI: 10.1016/j.ifacol.2015.11.154
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AltitudeFloorBarrier",
    "Barrier",
    "ConnectivityBarrier",
    "GeofenceBoxBarrier",
    "SafeDistanceBarrier",
    "SpeedLimitBarrier",
    "SphereObstacleBarrier",
]


@dataclass(frozen=True)
class BarrierRows:
    """Linear constraints ``A u ≤ b`` on the stacked acceleration command."""

    A: NDArray[np.floating]
    b: NDArray[np.floating]

    @staticmethod
    def empty(n_controls: int) -> "BarrierRows":
        return BarrierRows(np.zeros((0, n_controls)), np.zeros(0))


class Barrier(ABC):
    """A safe set ``h ≥ 0`` that can express itself as constraints on input."""

    name: str = "barrier"

    @abstractmethod
    def values(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Barrier value(s) at this state. Negative means already unsafe."""

    @abstractmethod
    def rows(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> BarrierRows:
        """Linear constraints the accelerations must satisfy."""

    def margin(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
    ) -> float:
        """Worst-case barrier value, the number to plot and to assert on."""
        vals = self.values(positions, velocities)
        return float(np.min(vals)) if np.size(vals) else float("inf")


def _second_order_rhs(h: float, h_dot: float, k1: float, k2: float) -> float:
    """Right-hand side of the high-order condition for linear class-K.

    With ``ψ₁ = ḣ + k₁h`` and ``ψ̇₁ + k₂ψ₁ ≥ 0`` the requirement expands to
    ``ḧ ≥ −(k₁ + k₂)ḣ − k₁k₂h``. Both gains must be positive: ``k₁`` sets how
    hard the barrier pushes back on approach, ``k₂`` how much overshoot of
    that push is tolerated.
    """
    return -(k1 + k2) * h_dot - k1 * k2 * h


class SafeDistanceBarrier(Barrier):
    """Keep every pair of vehicles at least *safe_distance* apart.

    Uses ``h = ‖Δp‖² − d²`` rather than ``‖Δp‖ − d``: the squared form is
    smooth everywhere, including the coincident case that the norm form
    cannot differentiate, and it is the coincident case a swap manoeuvre
    drives straight towards.

    The pair constraint couples both vehicles' accelerations, so a single QP
    over the whole fleet splits the avoidance effort between them. Solving
    per-vehicle instead makes each assume the other will not move, and both
    then dodge the same way.
    """

    name = "safe_distance"

    def __init__(self, safe_distance: float, k1: float = 2.0, k2: float = 2.0) -> None:
        if safe_distance <= 0.0:
            raise ValueError(f"safe_distance must be positive, got {safe_distance}")
        if k1 <= 0.0 or k2 <= 0.0:
            raise ValueError(f"class-K gains must be positive, got k1={k1}, k2={k2}")
        self.safe_distance = float(safe_distance)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def _pairs(self, n: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def values(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        n = len(p)
        pairs = self._pairs(n)
        if not pairs:
            return np.zeros(0)
        return np.array(
            [float(np.dot(p[i] - p[j], p[i] - p[j])) - self.safe_distance**2 for i, j in pairs]
        )

    def rows(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        n = len(p)
        pairs = self._pairs(n)
        if not pairs:
            return BarrierRows.empty(3 * n)

        A = np.zeros((len(pairs), 3 * n))
        b = np.zeros(len(pairs))
        for row, (i, j) in enumerate(pairs):
            dp = p[i] - p[j]
            dv = v[i] - v[j]
            h = float(dp @ dp) - self.safe_distance**2
            h_dot = 2.0 * float(dp @ dv)
            # ḧ = 2‖Δv‖² + 2 Δp·(a_i − a_j); the constant part moves to b.
            free = 2.0 * float(dv @ dv)
            A[row, 3 * i : 3 * i + 3] = -2.0 * dp
            A[row, 3 * j : 3 * j + 3] = 2.0 * dp
            b[row] = free - _second_order_rhs(h, h_dot, self.k1, self.k2)
        return BarrierRows(A, b)


class ConnectivityBarrier(Barrier):
    """Keep every pair of vehicles *within* ``comm_range`` of each other.

    The mirror image of :class:`SafeDistanceBarrier`, and worth having for
    exactly that reason: run the two together and the fleet is squeezed into
    an annulus, close enough to talk and far enough not to touch. It is also
    the honest way to show a filter becoming infeasible, since a safe
    distance larger than the communication range asks for the impossible.
    """

    name = "connectivity"

    def __init__(self, comm_range: float, k1: float = 2.0, k2: float = 2.0) -> None:
        if comm_range <= 0.0:
            raise ValueError(f"comm_range must be positive, got {comm_range}")
        self.comm_range = float(comm_range)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def values(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        n = len(p)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if not pairs:
            return np.zeros(0)
        return np.array(
            [self.comm_range**2 - float(np.dot(p[i] - p[j], p[i] - p[j])) for i, j in pairs]
        )

    def rows(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        n = len(p)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if not pairs:
            return BarrierRows.empty(3 * n)

        A = np.zeros((len(pairs), 3 * n))
        b = np.zeros(len(pairs))
        for row, (i, j) in enumerate(pairs):
            dp = p[i] - p[j]
            dv = v[i] - v[j]
            h = self.comm_range**2 - float(dp @ dp)
            h_dot = -2.0 * float(dp @ dv)
            free = -2.0 * float(dv @ dv)
            A[row, 3 * i : 3 * i + 3] = 2.0 * dp
            A[row, 3 * j : 3 * j + 3] = -2.0 * dp
            b[row] = free - _second_order_rhs(h, h_dot, self.k1, self.k2)
        return BarrierRows(A, b)


class SphereObstacleBarrier(Barrier):
    """Keep every vehicle outside a set of static spheres.

    Unlike the pairwise barrier the obstacle does not manoeuvre, so the whole
    avoidance burden is the vehicle's and the constraint touches one block of
    the decision vector.
    """

    name = "sphere_obstacle"

    def __init__(
        self,
        centres: NDArray[np.floating],
        radii: NDArray[np.floating] | float,
        clearance: float = 0.0,
        k1: float = 2.0,
        k2: float = 2.0,
    ) -> None:
        self.centres = np.atleast_2d(np.asarray(centres, dtype=float))
        r = np.asarray(radii, dtype=float)
        self.radii = np.full(len(self.centres), float(r)) if r.ndim == 0 else r.reshape(-1)
        if len(self.radii) != len(self.centres):
            raise ValueError("radii must be scalar or one per centre")
        self.clearance = float(clearance)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def _effective(self) -> NDArray[np.floating]:
        return self.radii + self.clearance

    def values(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        eff = self._effective()
        out = []
        for i in range(len(p)):
            for c, r in zip(self.centres, eff):
                d = p[i] - c
                out.append(float(d @ d) - r**2)
        return np.array(out) if out else np.zeros(0)

    def rows(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        n = len(p)
        eff = self._effective()
        rows_A, rows_b = [], []
        for i in range(n):
            for c, r in zip(self.centres, eff):
                dp = p[i] - c
                dv = v[i]
                h = float(dp @ dp) - r**2
                h_dot = 2.0 * float(dp @ dv)
                free = 2.0 * float(dv @ dv)
                row = np.zeros(3 * n)
                row[3 * i : 3 * i + 3] = -2.0 * dp
                rows_A.append(row)
                rows_b.append(free - _second_order_rhs(h, h_dot, self.k1, self.k2))
        if not rows_A:
            return BarrierRows.empty(3 * n)
        return BarrierRows(np.vstack(rows_A), np.asarray(rows_b))


class GeofenceBoxBarrier(Barrier):
    """Keep every vehicle inside an axis-aligned box.

    Each face is its own barrier with relative degree two, so the fence
    starts decelerating the vehicle in time rather than clamping the position
    once it has already crossed. A clamp is not a safety guarantee; it is a
    report that safety was already lost.
    """

    name = "geofence"

    def __init__(
        self,
        lower: NDArray[np.floating],
        upper: NDArray[np.floating],
        k1: float = 2.0,
        k2: float = 2.0,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float).reshape(3)
        self.upper = np.asarray(upper, dtype=float).reshape(3)
        if np.any(self.upper <= self.lower):
            raise ValueError(f"upper must exceed lower, got {self.lower} and {self.upper}")
        self.k1 = float(k1)
        self.k2 = float(k2)

    def values(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        return np.concatenate([(p - self.lower).ravel(), (self.upper - p).ravel()])

    def rows(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        n = len(p)
        rows_A, rows_b = [], []
        for i in range(n):
            for axis in range(3):
                for sign, bound in ((1.0, self.lower[axis]), (-1.0, self.upper[axis])):
                    # h = sign * (p - bound), so ḣ = sign * v and ḧ = sign * a.
                    h = sign * (p[i, axis] - bound)
                    h_dot = sign * v[i, axis]
                    row = np.zeros(3 * n)
                    row[3 * i + axis] = -sign
                    rows_A.append(row)
                    rows_b.append(-_second_order_rhs(h, h_dot, self.k1, self.k2))
        if not rows_A:
            return BarrierRows.empty(3 * n)
        return BarrierRows(np.vstack(rows_A), np.asarray(rows_b))


class AltitudeFloorBarrier(Barrier):
    """Keep every vehicle above a minimum altitude.

    A one-sided geofence, kept separate because ground contact is the
    failure this library's readers care about most and it deserves to be
    plottable on its own rather than buried among six fence faces.
    """

    name = "altitude_floor"

    def __init__(self, floor: float, k1: float = 2.0, k2: float = 2.0) -> None:
        self.floor = float(floor)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def values(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        return p[:, 2] - self.floor

    def rows(self, positions, velocities):
        p = np.asarray(positions, dtype=float)
        v = np.asarray(velocities, dtype=float)
        n = len(p)
        A = np.zeros((n, 3 * n))
        b = np.zeros(n)
        for i in range(n):
            h = p[i, 2] - self.floor
            h_dot = v[i, 2]
            A[i, 3 * i + 2] = -1.0
            b[i] = -_second_order_rhs(h, h_dot, self.k1, self.k2)
        return BarrierRows(A, b)


class SpeedLimitBarrier(Barrier):
    """Hold every vehicle below *max_speed*.

    The one barrier here with relative degree **one**: the constraint is on
    velocity, which acceleration moves directly, so it uses the plain
    condition ``ḣ ≥ −αh`` rather than the high-order form. Running it through
    the second-order machinery would be wrong -- and would look like it
    worked, because the extra derivative term is not identically zero.
    """

    name = "speed_limit"

    def __init__(self, max_speed: float, alpha: float = 4.0) -> None:
        if max_speed <= 0.0:
            raise ValueError(f"max_speed must be positive, got {max_speed}")
        self.max_speed = float(max_speed)
        self.alpha = float(alpha)

    def values(self, positions, velocities):
        v = np.asarray(velocities, dtype=float)
        return self.max_speed**2 - np.sum(v * v, axis=1)

    def rows(self, positions, velocities):
        v = np.asarray(velocities, dtype=float)
        n = len(v)
        A = np.zeros((n, 3 * n))
        b = np.zeros(n)
        for i in range(n):
            h = self.max_speed**2 - float(v[i] @ v[i])
            # ḣ = −2v·a, so the condition is 2v·a ≤ αh.
            A[i, 3 * i : 3 * i + 3] = 2.0 * v[i]
            b[i] = self.alpha * h
        return BarrierRows(A, b)
