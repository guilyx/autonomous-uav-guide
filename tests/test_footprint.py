# Erwin Lejeune - 2026-02-18
"""Tests for vehicle footprints and swarm envelopes."""

import numpy as np

from uav_sim.vehicles.footprint import (
    CircularFootprint,
    RectangularFootprint,
    swarm_convex_hull,
)


class TestCircularFootprint:
    def test_contains_centre(self):
        fp = CircularFootprint(radius=0.5)
        assert fp.contains_point(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 0.0)

    def test_outside(self):
        fp = CircularFootprint(radius=0.5)
        assert not fp.contains_point(np.array([2.0, 1.0]), np.array([1.0, 1.0]), 0.0)

    def test_bounding_radius(self):
        fp = CircularFootprint(radius=0.3)
        assert fp.bounding_radius() == 0.3

    def test_vertices_shape(self):
        fp = CircularFootprint(radius=1.0)
        v = fp.get_vertices(np.array([0.0, 0.0]), 0.0)
        assert v.shape == (16, 2)


class TestRectangularFootprint:
    def test_contains_centre(self):
        fp = RectangularFootprint(length=1.0, width=0.5)
        assert fp.contains_point(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0)

    def test_outside(self):
        fp = RectangularFootprint(length=1.0, width=0.5)
        assert not fp.contains_point(np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.0)

    def test_rotated(self):
        fp = RectangularFootprint(length=2.0, width=0.2)
        assert fp.contains_point(np.array([0.0, 0.8]), np.array([0.0, 0.0]), heading=np.pi / 2)


class TestSwarmConvexHull:
    def test_triangle(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        hull = swarm_convex_hull(pts)
        assert hull.shape[0] == 3

    def test_two_points_passthrough(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        hull = swarm_convex_hull(pts)
        assert hull.shape[0] == 2
