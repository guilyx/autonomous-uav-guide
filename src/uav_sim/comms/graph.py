# Erwin Lejeune - 2026-08-27
"""Graph metrics for a flying network, and their gradients.

The quantity that matters is the **algebraic connectivity** λ₂: the second
smallest eigenvalue of the weighted graph Laplacian. It is strictly
positive exactly while the network is connected, and it degrades smoothly
as links stretch -- so unlike "is the graph connected", which is a boolean
that tells a controller nothing until it is already too late, λ₂ can be
pushed on.

Its gradient with respect to position has a closed form through the Fiedler
vector, which is what makes connectivity maintenance a local controller
rather than a global optimisation:

    ∂λ₂/∂p_i = Σ_j (∂w_ij/∂p_i)(v_i − v_j)²

The Fiedler vector ``v`` is the eigenvector for λ₂, and it is worth reading
that formula for what it says: effort goes where the *Fiedler vector*
disagrees most, not where the distance is greatest. Those are different
edges. The Fiedler vector is near-constant within a tightly connected
cluster and jumps across the weak cut between clusters, so the gradient
concentrates on the links actually holding the network together and
ignores redundant ones inside a clump.

References:
- M. Fiedler, "Algebraic connectivity of graphs," Czechoslovak Mathematical
  Journal, 1973.
- M. C. De Gennaro & A. Jadbabaie, "Decentralized Control of Connectivity
  for Multi-Agent Systems," IEEE CDC, 2006. DOI: 10.1109/CDC.2006.377041
- L. Sabattini et al., "Decentralized connectivity maintenance for
  cooperative control of mobile robotic systems," IJRR, 2013.
  DOI: 10.1177/0278364913499085
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .radio import LinkModel

__all__ = [
    "algebraic_connectivity",
    "connectivity_gradient",
    "degree_of_connectivity",
    "fiedler_vector",
    "hop_counts",
    "laplacian",
]


def laplacian(weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """Weighted graph Laplacian ``L = D − W``."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"weights must be square, got {w.shape}")
    return np.diag(w.sum(axis=1)) - w


def algebraic_connectivity(weights: NDArray[np.floating]) -> float:
    """λ₂ of the Laplacian: positive exactly while the graph is connected."""
    w = np.asarray(weights, dtype=float)
    if w.shape[0] < 2:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(laplacian(w))
    return float(eigenvalues[1])


def fiedler_vector(weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """Eigenvector of λ₂, normalised. Zero vector for a trivial graph."""
    w = np.asarray(weights, dtype=float)
    if w.shape[0] < 2:
        return np.zeros(w.shape[0])
    _, vectors = np.linalg.eigh(laplacian(w))
    return vectors[:, 1]


def connectivity_gradient(
    positions: NDArray[np.floating],
    link: LinkModel,
) -> NDArray[np.floating]:
    """``∂λ₂/∂p`` for every agent, shape ``(N, 3)``.

    Ascending this increases algebraic connectivity. The result is exact for
    a simple λ₂ (no repeated eigenvalue); a perfectly symmetric formation
    can produce a repeated λ₂, where the eigenvector -- and so this gradient
    -- is not unique. In practice the symmetry is broken by any disturbance,
    but it is the reason a connectivity controller can look briefly erratic
    in a perfectly regular lattice.
    """
    p = np.asarray(positions, dtype=float)
    n = len(p)
    grad = np.zeros_like(p)
    if n < 2:
        return grad

    w = link.weights(p)
    v = fiedler_vector(w)

    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    safe = np.maximum(dist, 1e-9)
    dw_dd = link.dweight_ddistance(dist)
    np.fill_diagonal(dw_dd, 0.0)

    # ∂w_ij/∂p_i = (dw/dd)(p_i − p_j)/d, weighted by the Fiedler disagreement.
    disagreement = (v[:, None] - v[None, :]) ** 2
    coeff = dw_dd * disagreement / safe
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum("ij,ijk->ik", coeff, diff)
    return grad


def degree_of_connectivity(
    weights: NDArray[np.floating],
    threshold: float = 0.5,
) -> int:
    """How many agents can be lost before the network splits.

    Computed by brute-force removal, which is fine at swarm sizes and exact,
    unlike the usual spectral proxies. A value of 1 means the network is a
    chain or a tree: every agent is a single point of failure. Two or more
    means there is a redundant path, which is what a mission that cannot
    tolerate one radio dying actually needs.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    if n < 2:
        return 0
    adjacency = w > threshold
    if not _connected(adjacency):
        return 0
    for k in range(1, n - 1):
        if _splits_after_removing(adjacency, k):
            return k
    return n - 1


def _connected(adjacency: NDArray[np.bool_]) -> bool:
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for nxt in np.flatnonzero(adjacency[node] & ~seen):
            seen[nxt] = True
            stack.append(int(nxt))
    return bool(seen.all())


def _splits_after_removing(adjacency: NDArray[np.bool_], k: int) -> bool:
    from itertools import combinations

    n = adjacency.shape[0]
    for victims in combinations(range(n), k):
        keep = np.setdiff1d(np.arange(n), np.asarray(victims))
        if len(keep) > 1 and not _connected(adjacency[np.ix_(keep, keep)]):
            return True
    return False


def hop_counts(
    weights: NDArray[np.floating],
    source: int = 0,
    threshold: float = 0.5,
) -> NDArray[np.floating]:
    """Hops from *source* to every agent; ``inf`` where unreachable.

    The relay problem is about *reachability to the base*, not about the
    network being connected in the abstract, so this is the metric a relay
    controller and its plots want.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[0]
    adjacency = w > threshold
    hops = np.full(n, np.inf)
    hops[source] = 0.0
    frontier = [source]
    while frontier:
        nxt: list[int] = []
        for node in frontier:
            for cand in np.flatnonzero(adjacency[node] & np.isinf(hops)):
                hops[cand] = hops[node] + 1.0
                nxt.append(int(cand))
        frontier = nxt
    return hops
