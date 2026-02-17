# Erwin Lejeune - 2026-02-17
"""Geometric obstacle primitives for collision checking and visualisation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


class Obstacle(ABC):
    """Abstract obstacle with collision query."""

    @abstractmethod
    def contains(self, point: NDArray[np.floating]) -> bool:
        """Return True if *point* lies inside the obstacle."""
        ...

    @abstractmethod
    def distance(self, point: NDArray[np.floating]) -> float:
        """Signed distance from *point* to obstacle surface (negative = inside)."""
        ...

    @abstractmethod
    def bounding_box(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return (min_corner, max_corner) of the AABB."""
        ...


@dataclass
class SphereObstacle(Obstacle):
    """Sphere obstacle defined by centre and radius."""

    centre: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0

    def contains(self, point: NDArray[np.floating]) -> bool:
        return float(np.linalg.norm(point - self.centre)) <= self.radius

    def distance(self, point: NDArray[np.floating]) -> float:
        return float(np.linalg.norm(point - self.centre)) - self.radius

    def bounding_box(self):
        return self.centre - self.radius, self.centre + self.radius


@dataclass
class BoxObstacle(Obstacle):
    """Axis-aligned box obstacle."""

    min_corner: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    max_corner: NDArray[np.floating] = field(default_factory=lambda: np.ones(3))

    def contains(self, point: NDArray[np.floating]) -> bool:
        return bool(
            np.all(point >= self.min_corner) and np.all(point <= self.max_corner)
        )

    def distance(self, point: NDArray[np.floating]) -> float:
        clamped = np.clip(point, self.min_corner, self.max_corner)
        return float(np.linalg.norm(point - clamped))

    def bounding_box(self):
        return np.array(self.min_corner), np.array(self.max_corner)


@dataclass
class CylinderObstacle(Obstacle):
    """Vertical cylinder obstacle."""

    centre: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0
    height: float = 5.0

    def contains(self, point: NDArray[np.floating]) -> bool:
        dxy = np.linalg.norm(point[:2] - self.centre[:2])
        dz = point[2] - self.centre[2]
        return bool(dxy <= self.radius and 0 <= dz <= self.height)

    def distance(self, point: NDArray[np.floating]) -> float:
        dxy = float(np.linalg.norm(point[:2] - self.centre[:2])) - self.radius
        dz_lo = self.centre[2] - point[2]
        dz_hi = point[2] - (self.centre[2] + self.height)
        return max(dxy, dz_lo, dz_hi)

    def bounding_box(self):
        lo = np.array(
            [self.centre[0] - self.radius, self.centre[1] - self.radius, self.centre[2]]
        )
        hi = np.array(
            [
                self.centre[0] + self.radius,
                self.centre[1] + self.radius,
                self.centre[2] + self.height,
            ]
        )
        return lo, hi
