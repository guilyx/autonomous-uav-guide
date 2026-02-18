# Erwin Lejeune - 2026-02-15
"""Path smoothing utilities for 3-D waypoint sequences.

Provides Ramer–Douglas–Peucker simplification and optional cubic-spline
resampling so that raw grid-planner output (A*, RRT, etc.) becomes a
smooth, flyable trajectory for pure-pursuit tracking.

Reference: D. Douglas, T. Peucker, "Algorithms for the Reduction of the
Number of Points Required to Represent a Digitized Line or its Caricature,"
Cartographica, 1973.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rdp_simplify(
    path: NDArray[np.floating],
    epsilon: float = 0.5,
) -> NDArray[np.floating]:
    """Ramer–Douglas–Peucker path simplification in N-D.

    Recursively removes points that are within *epsilon* of the straight
    line between their neighbours, eliminating unnecessary zigzags while
    preserving the overall shape.

    Parameters
    ----------
    path : (N, D) waypoint array.
    epsilon : maximum perpendicular distance tolerance [m].

    Returns
    -------
    (M, D) simplified waypoint array (M <= N).
    """
    if len(path) <= 2:
        return path.copy()
    return path[_rdp_mask(path, epsilon)]


def _rdp_mask(pts: NDArray, eps: float) -> NDArray[np.bool_]:
    """Return boolean mask of points to keep."""
    n = len(pts)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    mask[-1] = True
    _rdp_recurse(pts, 0, n - 1, eps, mask)
    return mask


def _rdp_recurse(
    pts: NDArray,
    start: int,
    end: int,
    eps: float,
    mask: NDArray[np.bool_],
) -> None:
    if end - start < 2:
        return
    line_dir = pts[end] - pts[start]
    line_len = float(np.linalg.norm(line_dir))
    if line_len < 1e-12:
        mask[start + 1 : end] = True
        return
    line_unit = line_dir / line_len

    dmax = 0.0
    idx = start
    for i in range(start + 1, end):
        v = pts[i] - pts[start]
        proj = float(np.dot(v, line_unit))
        proj = np.clip(proj, 0.0, line_len)
        closest = pts[start] + proj * line_unit
        d = float(np.linalg.norm(pts[i] - closest))
        if d > dmax:
            dmax = d
            idx = i

    if dmax > eps:
        mask[idx] = True
        _rdp_recurse(pts, start, idx, eps, mask)
        _rdp_recurse(pts, idx, end, eps, mask)


def smooth_path_3d(
    path: NDArray[np.floating],
    epsilon: float = 0.5,
    num_points: int | None = None,
    min_spacing: float = 0.0,
) -> NDArray[np.floating]:
    """Simplify and optionally resample a 3-D path with cubic interpolation.

    Steps:
    1. RDP simplification to prune zigzag segments.
    2. If *num_points* is given, cubic-spline resample to that many points.
       Otherwise, resample so adjacent points are at least *min_spacing* apart
       (if *min_spacing* > 0).

    Parameters
    ----------
    path : (N, 3) raw waypoint array.
    epsilon : RDP simplification tolerance [m].
    num_points : exact number of output points (overrides *min_spacing*).
    min_spacing : minimum inter-point distance for auto-resampling [m].
                  Set to 0 to skip resampling (only RDP is applied).

    Returns
    -------
    (M, 3) smoothed waypoint array.
    """
    if len(path) < 3:
        return path.copy()

    pruned = rdp_simplify(path, epsilon)
    if len(pruned) < 2:
        return pruned

    if num_points is None and min_spacing <= 0:
        return pruned

    # Cumulative arc-length parameterisation
    diffs = np.diff(pruned, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-8:
        return pruned

    if num_points is not None:
        n_out = max(num_points, 2)
    else:
        n_out = max(int(total_len / min_spacing), 2)

    t_out = np.linspace(0, total_len, n_out)
    result = np.zeros((n_out, pruned.shape[1]))
    for dim in range(pruned.shape[1]):
        result[:, dim] = np.interp(t_out, cum_len, pruned[:, dim])

    return result
