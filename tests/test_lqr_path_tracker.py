# Erwin Lejeune - 2026-02-15
"""Tests for LQR path tracker controller."""

import numpy as np

from uav_sim.path_tracking.lqr_path_tracker import LQRPathTracker


class TestLQRPathTracker:
    def test_basic_tracking(self):
        tracker = LQRPathTracker(lookahead=2.0, speed=1.0)
        state = np.zeros(12)
        state[2] = 5.0
        path = np.array([[0, 0, 5], [5, 0, 5], [10, 0, 5.0]])
        wrench = tracker.compute(state, path)
        assert wrench.shape == (4,)
        assert wrench[0] > 0

    def test_path_complete(self):
        tracker = LQRPathTracker()
        path = np.array([[0, 0, 5], [5, 0, 5.0]])
        assert tracker.is_path_complete(np.array([4.5, 0.0, 5.0]), path, threshold=1.0)
        assert not tracker.is_path_complete(np.array([0.0, 0.0, 5.0]), path, threshold=1.0)

    def test_reset(self):
        tracker = LQRPathTracker()
        tracker._idx = 5
        tracker.reset()
        assert tracker._idx == 0

    def test_empty_path(self):
        tracker = LQRPathTracker()
        state = np.zeros(12)
        wrench = tracker.compute(state, np.empty((0, 3)))
        assert wrench.shape == (4,)
