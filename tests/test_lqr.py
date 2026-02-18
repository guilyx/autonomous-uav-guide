# Erwin Lejeune - 2026-02-16
"""Tests for the LQR controller."""

import numpy as np
import pytest

from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class TestLQRInit:
    def test_gain_shape(self):
        lqr = LQRController()
        assert lqr.K.shape == (4, 12)

    def test_hover_wrench(self):
        lqr = LQRController()
        assert pytest.approx(lqr.hover_wrench[0], rel=1e-6) == 1.5 * 9.81


class TestLQRCompute:
    def test_at_equilibrium_returns_hover(self):
        lqr = LQRController()
        wrench = lqr.compute(np.zeros(12))
        np.testing.assert_allclose(wrench, lqr.hover_wrench, atol=1e-6)

    def test_below_target_increases_thrust(self):
        lqr = LQRController()
        state = np.zeros(12)
        state[2] = -0.5  # below origin
        wrench = lqr.compute(state)
        assert wrench[0] > lqr.hover_wrench[0]

    def test_target_state_offset(self):
        lqr = LQRController()
        target = np.zeros(12)
        target[2] = 1.0  # target at z=1
        state = np.zeros(12)
        state[2] = 1.0  # at target
        wrench = lqr.compute(state, target_state=target)
        np.testing.assert_allclose(wrench, lqr.hover_wrench, atol=1e-6)


class TestLQRClosedLoop:
    def test_hover_convergence(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.5]))
        hover_force = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_force))

        lqr = LQRController()
        target = np.zeros(12)
        target[2] = 1.0

        dt = 0.001
        for _ in range(5000):
            wrench = lqr.compute(quad.state, target_state=target)
            quad.step(wrench, dt)

        assert abs(quad.position[2] - 1.0) < 0.15
