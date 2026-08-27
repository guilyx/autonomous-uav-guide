# Erwin Lejeune - 2026-08-27
"""The quadratic program behind every CBF safety filter.

A control barrier function turns "stay safe" into a set of linear
inequalities on the control input. Enforcing them while staying as close as
possible to what the nominal controller asked for is exactly a projection
onto a polyhedron::

    minimise  ½‖u − u_nom‖²   subject to   A u ≤ b

which is a *least-distance program*. Lawson and Hanson showed an LDP is the
dual of a non-negative least-squares problem, so it is solved exactly, in
finitely many steps, by an NNLS routine -- no iteration limit to tune and no
interior-point solver to depend on. That matters here: this runs inside the
control loop, and a solver that merely *usually* converges is not a safety
guarantee.

Reference: C. L. Lawson & R. J. Hanson, "Solving Least Squares Problems",
Prentice-Hall, 1974, chapter 23 (Least Distance Programming).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


class InfeasibleQPError(RuntimeError):
    """Raised when no control input satisfies every barrier constraint.

    This is a real and meaningful outcome, not a numerical hiccup. It means
    the state has already reached a point where the constraints conflict --
    typically because two barriers demand opposite accelerations, or because
    the vehicle entered the region too fast for any admissible input to stop
    it in time. Callers should treat it as a safety event, not retry.
    """


def solve_least_distance(
    G: NDArray[np.floating],
    h: NDArray[np.floating],
    *,
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """Solve ``min ‖z‖`` subject to ``G z ≥ h``.

    Args:
        G: ``(m, n)`` constraint matrix.
        h: ``(m,)`` right-hand side.
        tol: Residual norm below which the dual is judged degenerate, which
            is how the algorithm reports an empty feasible set.

    Returns:
        ``(n,)`` the point of the feasible polyhedron closest to the origin.

    Raises:
        InfeasibleQPError: if ``{z : G z ≥ h}`` is empty.
    """
    G = np.atleast_2d(np.asarray(G, dtype=float))
    h = np.asarray(h, dtype=float).reshape(-1)
    if G.shape[0] != h.shape[0]:
        raise ValueError(f"G has {G.shape[0]} rows but h has {h.shape[0]} entries")

    n = G.shape[1]
    if G.shape[0] == 0:
        return np.zeros(n)

    # Lawson-Hanson: stack the constraints into E and solve the NNLS dual.
    E = np.vstack([G.T, h.reshape(1, -1)])
    f = np.zeros(n + 1)
    f[n] = 1.0

    lam, _ = nnls(E, f)
    residual = E @ lam - f

    # A vanishing residual means the dual found no separating direction,
    # which is precisely the certificate that the primal is infeasible.
    if float(np.linalg.norm(residual)) <= tol:
        raise InfeasibleQPError("no z satisfies G z >= h")

    return -residual[:n] / residual[n]


def solve_safety_qp(
    u_nominal: NDArray[np.floating],
    A: NDArray[np.floating] | None,
    b: NDArray[np.floating] | None,
    *,
    u_min: NDArray[np.floating] | float | None = None,
    u_max: NDArray[np.floating] | float | None = None,
) -> NDArray[np.floating]:
    """Smallest change to *u_nominal* that satisfies ``A u ≤ b``.

    Args:
        u_nominal: ``(n,)`` what the nominal controller asked for.
        A: ``(m, n)`` barrier constraint matrix, or ``None`` for no
            constraints.
        b: ``(m,)`` barrier constraint bounds.
        u_min: Lower actuator bound, scalar or ``(n,)``. Folded into the same
            QP rather than clipped afterwards -- clipping a solved input can
            walk straight back through a barrier it was chosen to respect.
        u_max: Upper actuator bound.

    Returns:
        ``(n,)`` the filtered control input.

    Raises:
        InfeasibleQPError: if the constraints admit no input.
    """
    u_nominal = np.asarray(u_nominal, dtype=float).reshape(-1)
    n = u_nominal.size

    rows: list[NDArray[np.floating]] = []
    rhs: list[float] = []

    if A is not None and b is not None:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.asarray(b, dtype=float).reshape(-1)
        if A.size:
            if A.shape[1] != n:
                raise ValueError(f"A has {A.shape[1]} columns but u has {n} entries")
            rows.extend(A)
            rhs.extend(b)

    def _bound(value: NDArray[np.floating] | float) -> NDArray[np.floating]:
        arr = np.asarray(value, dtype=float)
        return np.full(n, float(arr)) if arr.ndim == 0 else arr.reshape(-1)

    if u_max is not None:
        hi = _bound(u_max)
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            rhs.append(hi[i])
    if u_min is not None:
        lo = _bound(u_min)
        for i in range(n):
            row = np.zeros(n)
            row[i] = -1.0
            rows.append(row)
            rhs.append(-lo[i])

    if not rows:
        return u_nominal.copy()

    A_all = np.vstack(rows)
    b_all = np.asarray(rhs, dtype=float)

    # Shift to z = u - u_nominal so the objective is a plain distance to the
    # origin, then flip the sense for the >= form the LDP routine expects.
    z = solve_least_distance(-A_all, A_all @ u_nominal - b_all)
    return u_nominal + z
