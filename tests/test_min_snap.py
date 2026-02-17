# Erwin Lejeune - 2026-02-16
"""Tests for min-snap and polynomial trajectory generation."""

import numpy as np
import pytest

from quadrotor_sim.planning.min_snap import MinSnapTrajectory
from quadrotor_sim.planning.polynomial_trajectory import PolynomialTrajectory

# ---------------------------------------------------------------------------
# Polynomial trajectory
# ---------------------------------------------------------------------------


class TestPolynomialTrajectory:
    def test_quintic_starts_and_ends_at_waypoints(self):
        traj = PolynomialTrajectory(order=5)
        waypoints = np.array([[0, 0, 0], [1, 2, 1], [3, 1, 2.0]])
        seg_times = np.array([1.0, 1.0])
        coeffs = traj.generate(waypoints, seg_times)

        # Check start of first segment.
        for d in range(3):
            assert (
                pytest.approx(traj._poly_eval(coeffs[d][0], 0.0), abs=1e-6)
                == waypoints[0, d]
            )

        # Check end of first segment = waypoint 1.
        for d in range(3):
            assert (
                pytest.approx(traj._poly_eval(coeffs[d][0], 1.0), abs=1e-6)
                == waypoints[1, d]
            )

    def test_evaluate_produces_correct_shape(self):
        traj = PolynomialTrajectory()
        waypoints = np.array([[0, 0, 0], [1, 1, 1.0]])
        seg_times = np.array([2.0])
        coeffs = traj.generate(waypoints, seg_times)
        times, positions = traj.evaluate(coeffs, seg_times, dt=0.1)
        assert len(times) == len(positions)
        assert positions.shape[1] == 3

    def test_low_order_raises(self):
        with pytest.raises(ValueError):
            PolynomialTrajectory(order=2)


# ---------------------------------------------------------------------------
# Min-snap trajectory
# ---------------------------------------------------------------------------


class TestMinSnapTrajectory:
    def test_passes_through_waypoints(self):
        ms = MinSnapTrajectory()
        waypoints = np.array([[0, 0, 0], [1, 0, 1], [2, 1, 0.5]])
        seg_times = np.array([1.0, 1.0])
        coeffs = ms.generate(waypoints, seg_times)

        # Start of first segment should be waypoint 0.
        for d in range(3):
            val = float(coeffs[d][0, 0])  # c0 = p(0)
            assert pytest.approx(val, abs=0.1) == waypoints[0, d]

    def test_zero_velocity_at_endpoints(self):
        ms = MinSnapTrajectory()
        waypoints = np.array([[0, 0, 0], [5, 0, 0.0]])
        seg_times = np.array([2.0])
        coeffs = ms.generate(waypoints, seg_times)

        # Velocity at t=0 is c1.
        for d in range(3):
            assert pytest.approx(float(coeffs[d][0, 1]), abs=0.1) == 0.0

    def test_evaluate_shape(self):
        ms = MinSnapTrajectory()
        waypoints = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2.0]])
        seg_times = np.array([1.0, 1.5])
        coeffs = ms.generate(waypoints, seg_times)
        times, positions = ms.evaluate(coeffs, seg_times, dt=0.05)
        assert len(times) == len(positions)
        assert positions.shape[1] == 3
        assert len(times) > 0

    def test_smooth_trajectory(self):
        """Positions should change smoothly (no large jumps between consecutive points)."""
        ms = MinSnapTrajectory()
        waypoints = np.array([[0, 0, 0], [2, 0, 1], [4, 2, 0.5], [6, 0, 0.0]])
        seg_times = np.array([1.5, 1.5, 1.5])
        coeffs = ms.generate(waypoints, seg_times)
        _, positions = ms.evaluate(coeffs, seg_times, dt=0.01)
        diffs = np.diff(positions, axis=0)
        max_jump = np.max(np.linalg.norm(diffs, axis=1))
        assert max_jump < 0.5  # no discontinuities
