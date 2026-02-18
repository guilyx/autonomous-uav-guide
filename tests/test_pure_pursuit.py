# Erwin Lejeune - 2026-02-17
"""Tests for the 3D pure pursuit path tracker."""

from __future__ import annotations

import numpy as np

from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D


class TestPurePursuit3D:
    """Unit tests for PurePursuit3D."""

    def test_basic_tracking(self):
        """Tracker should advance along a simple straight path."""
        pp = PurePursuit3D(lookahead=1.0, waypoint_threshold=0.3, adaptive=False)
        path = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        pos = np.array([0.0, 0.0, 0.0])
        target = pp.compute_target(pos, path)
        assert target[0] > pos[0], "Target should be ahead on x-axis"

    def test_path_complete(self):
        """Should detect when close to final waypoint."""
        pp = PurePursuit3D(lookahead=1.0, waypoint_threshold=0.5)
        path = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        assert not pp.is_path_complete(np.array([0.0, 0.0, 0.0]), path)
        assert pp.is_path_complete(np.array([0.8, 0.0, 0.0]), path)

    def test_empty_path(self):
        """Should return current position for empty path."""
        pp = PurePursuit3D()
        pos = np.array([1.0, 2.0, 3.0])
        target = pp.compute_target(pos, np.zeros((0, 3)))
        np.testing.assert_array_equal(target, pos)

    def test_reset(self):
        """Reset should set waypoint index to zero."""
        pp = PurePursuit3D(lookahead=0.5, waypoint_threshold=0.5)
        path = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        # Place at first waypoint to advance the index
        pp.compute_target(np.array([0.0, 0, 0.0]), path)
        assert pp.current_index > 0
        pp.reset()
        assert pp.current_index == 0

    def test_adaptive_lookahead(self):
        """Adaptive mode should increase lookahead with speed."""
        pp = PurePursuit3D(lookahead=1.0, waypoint_threshold=0.3, adaptive=True)
        path = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        pos = np.array([0.0, 0.0, 0.0])
        slow = pp.compute_target(pos, path, velocity=np.array([0.1, 0.0, 0.0]))
        pp.reset()
        fast = pp.compute_target(pos, path, velocity=np.array([5.0, 0.0, 0.0]))
        assert fast[0] >= slow[0], "Higher speed should push target further"

    def test_sphere_segment_intersection(self):
        """Intersection helper should return correct results."""
        center = np.array([0.0, 0.0, 0.0])
        p1 = np.array([-2.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        pt = PurePursuit3D._intersect_sphere_segment(center, 1.0, p1, p2)
        assert pt is not None
        np.testing.assert_almost_equal(pt[0], 1.0, decimal=5)

    def test_no_intersection(self):
        """Should return None when sphere doesn't intersect segment."""
        center = np.array([0.0, 5.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        assert PurePursuit3D._intersect_sphere_segment(center, 1.0, p1, p2) is None
