# Erwin Lejeune - 2026-02-18
"""Vehicle footprint / envelope representations for costmap inflation.

Inspired by the ``simple_autonomous_car`` footprint module. Provides
circular and rectangular footprints that can be used to inflate
costmaps, check collisions, and draw swarm envelopes.

Usage::

    fp = CircularFootprint(radius=0.3)
    if fp.contains_point(obstacle_xy, drone_xy, heading=0.0):
        print("collision!")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class BaseFootprint(ABC):
    """Abstract 2D footprint for collision checking and costmap inflation."""

    @abstractmethod
    def get_vertices(self, position: NDArray[np.floating], heading: float) -> NDArray[np.floating]:
        """Return (N, 2) polygon vertices in world frame."""

    @abstractmethod
    def bounding_radius(self) -> float:
        """Circle that fully encloses the footprint."""

    @abstractmethod
    def contains_point(
        self, point: NDArray[np.floating], position: NDArray[np.floating], heading: float
    ) -> bool:
        """True if *point* lies inside the footprint centred at *position*."""

    def inflation_radius(self, padding: float = 0.0) -> float:
        return self.bounding_radius() + padding


@dataclass
class CircularFootprint(BaseFootprint):
    """Circle centred on the vehicle — typical for multirotors."""

    radius: float = 0.3

    def get_vertices(
        self,
        position: NDArray[np.floating],
        heading: float,  # noqa: ARG002
    ) -> NDArray[np.floating]:
        angles = np.linspace(0, 2 * np.pi, 17)[:-1]
        return np.column_stack(
            [
                position[0] + self.radius * np.cos(angles),
                position[1] + self.radius * np.sin(angles),
            ]
        )

    def bounding_radius(self) -> float:
        return self.radius

    def contains_point(
        self,
        point: NDArray[np.floating],
        position: NDArray[np.floating],
        heading: float,  # noqa: ARG002
    ) -> bool:
        return float(np.linalg.norm(point[:2] - position[:2])) <= self.radius


@dataclass
class RectangularFootprint(BaseFootprint):
    """Axis-aligned box that rotates with heading — useful for fixed-wing."""

    length: float = 0.5
    width: float = 0.3

    def get_vertices(self, position: NDArray[np.floating], heading: float) -> NDArray[np.floating]:
        hl, hw = self.length / 2, self.width / 2
        corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ corners.T).T + position[:2]

    def bounding_radius(self) -> float:
        return float(np.hypot(self.length / 2, self.width / 2))

    def contains_point(
        self,
        point: NDArray[np.floating],
        position: NDArray[np.floating],
        heading: float,
    ) -> bool:
        c, s = np.cos(-heading), np.sin(-heading)
        rot = np.array([[c, -s], [s, c]])
        local = rot @ (point[:2] - position[:2])
        return bool(abs(local[0]) <= self.length / 2 and abs(local[1]) <= self.width / 2)


def swarm_convex_hull(
    positions: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the 2D convex hull envelope of a swarm.

    Parameters
    ----------
    positions : (N, 2) or (N, 3) array of agent positions.

    Returns
    -------
    (M, 2) ordered hull vertices suitable for polygon plotting.
    """
    from scipy.spatial import ConvexHull

    pts = positions[:, :2]
    if len(pts) < 3:
        return pts
    hull = ConvexHull(pts)
    return pts[hull.vertices]
