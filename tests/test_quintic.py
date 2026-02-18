# Erwin Lejeune - 2026-02-15
"""Tests for quintic polynomial planner."""

import numpy as np

from uav_sim.trajectory_planning.quintic_polynomial import (
    QuinticPolynomialPlanner,
    QuinticState,
)


class TestQuinticPolynomial:
    def test_boundary_conditions(self):
        planner = QuinticPolynomialPlanner()
        start = QuinticState(
            pos=np.array([0.0, 0.0, 0.0]),
            vel=np.array([1.0, 0.0, 0.0]),
            acc=np.zeros(3),
        )
        goal = QuinticState(
            pos=np.array([10.0, 5.0, 3.0]),
            vel=np.array([0.0, 1.0, 0.0]),
            acc=np.zeros(3),
        )
        T = 5.0
        coeffs = planner.generate(start, goal, T)
        ts, pos, vel, acc = planner.evaluate(coeffs, T, dt=0.01)

        np.testing.assert_allclose(pos[0], start.pos, atol=1e-6)
        np.testing.assert_allclose(pos[-1], goal.pos, atol=0.1)
        np.testing.assert_allclose(vel[0], start.vel, atol=1e-6)

    def test_zero_velocity_rest_to_rest(self):
        planner = QuinticPolynomialPlanner()
        start = QuinticState(pos=np.array([0.0]), vel=np.array([0.0]), acc=np.array([0.0]))
        goal = QuinticState(pos=np.array([10.0]), vel=np.array([0.0]), acc=np.array([0.0]))
        coeffs = planner.generate(start, goal, T=4.0)
        _, pos, vel, _ = planner.evaluate(coeffs, T=4.0, dt=0.01)
        np.testing.assert_allclose(pos[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(pos[-1, 0], 10.0, atol=0.1)
        np.testing.assert_allclose(vel[0, 0], 0.0, atol=1e-6)

    def test_3d_trajectory(self):
        planner = QuinticPolynomialPlanner()
        start = QuinticState(pos=np.array([1, 2, 3.0]), vel=np.zeros(3), acc=np.zeros(3))
        goal = QuinticState(pos=np.array([10, 8, 6.0]), vel=np.zeros(3), acc=np.zeros(3))
        coeffs = planner.generate(start, goal, T=3.0)
        assert coeffs.shape == (3, 6)
        ts, pos, _, _ = planner.evaluate(coeffs, T=3.0, dt=0.1)
        assert len(ts) > 0
        assert pos.shape[1] == 3
