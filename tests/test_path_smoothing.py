# Erwin Lejeune - 2026-02-15
"""Tests for the path smoothing utilities."""

from __future__ import annotations

import numpy as np

from uav_sim.path_tracking.path_smoothing import rdp_simplify, smooth_path_3d


class TestRDPSimplify:
    def test_straight_line_collapses(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        result = rdp_simplify(pts, epsilon=0.1)
        assert len(result) == 2
        np.testing.assert_allclose(result[0], pts[0])
        np.testing.assert_allclose(result[-1], pts[-1])

    def test_zigzag_preserved_above_epsilon(self):
        pts = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0], [3, 2, 0]], dtype=float)
        result = rdp_simplify(pts, epsilon=0.5)
        assert len(result) == len(pts)

    def test_short_path_unchanged(self):
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        result = rdp_simplify(pts, epsilon=5.0)
        assert len(result) == 2

    def test_endpoints_always_kept(self):
        pts = np.linspace([0, 0, 0], [10, 0, 0], 50)
        result = rdp_simplify(pts, epsilon=0.01)
        np.testing.assert_allclose(result[0], pts[0])
        np.testing.assert_allclose(result[-1], pts[-1])


class TestSmoothPath3D:
    def test_resampling_with_num_points(self):
        pts = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        result = smooth_path_3d(pts, epsilon=0.1, num_points=20)
        assert len(result) == 20
        np.testing.assert_allclose(result[0], pts[0], atol=1e-6)
        np.testing.assert_allclose(result[-1], pts[-1], atol=1e-6)

    def test_resampling_with_min_spacing(self):
        pts = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        result = smooth_path_3d(pts, epsilon=0.1, min_spacing=2.0)
        assert len(result) >= 5
        diffs = np.linalg.norm(np.diff(result, axis=0), axis=1)
        np.testing.assert_allclose(diffs, diffs[0], atol=0.1)

    def test_no_resample_returns_pruned(self):
        pts = np.array(
            [[0, 0, 0], [0.1, 0.0001, 0], [0.2, 0, 0], [5, 5, 5]],
            dtype=float,
        )
        result = smooth_path_3d(pts, epsilon=0.5, min_spacing=0)
        assert len(result) <= len(pts)

    def test_too_short_path(self):
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        result = smooth_path_3d(pts, epsilon=0.5, num_points=10)
        assert len(result) == 2

    def test_altitude_preserved(self):
        pts = np.array([[0, 0, 5], [1, 0, 5], [2, 1, 5], [3, 1, 5]], dtype=float)
        result = smooth_path_3d(pts, epsilon=0.1, num_points=10)
        np.testing.assert_allclose(result[:, 2], 5.0, atol=1e-6)
