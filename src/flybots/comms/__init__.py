# Erwin Lejeune - 2026-08-27
"""Communication-aware swarming: the radio network as part of the plant.

A link model turns range into a smooth weight, the graph module turns those
weights into algebraic connectivity and its gradient, and the controllers
use that gradient either to keep a mesh alive under a task or to trade
coverage against reachability in a relay chain.
"""

from .controllers import ConnectivityController, RelayCoverageController
from .graph import (
    algebraic_connectivity,
    connectivity_gradient,
    degree_of_connectivity,
    fiedler_vector,
    hop_counts,
    laplacian,
)
from .radio import GaussianLink, LinkModel, PathLossLink

__all__ = [
    "ConnectivityController",
    "GaussianLink",
    "LinkModel",
    "PathLossLink",
    "RelayCoverageController",
    "algebraic_connectivity",
    "connectivity_gradient",
    "degree_of_connectivity",
    "fiedler_vector",
    "hop_counts",
    "laplacian",
]
