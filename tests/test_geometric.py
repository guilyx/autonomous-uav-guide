# Erwin Lejeune - 2026-02-16
"""Tests for the geometric SO(3) controller."""

import numpy as np
import pytest

from quadrotor_sim.control.geometric_controller import GeometricController
from quadrotor_sim.models.quadrotor import Quadrotor


class TestGeometricCompute:
    def test_hover_thrust_magnitude(self):
        ctrl = GeometricController()
        state = np.zeros(12)
        state[2] = 1.0
        wrench = ctrl.compute(state, target_pos=np.array([0.0, 0.0, 1.0]))
        expected = ctrl.config.mass * ctrl.config.gravity
        assert pytest.approx(wrench[0], rel=0.2) == expected

    def test_below_target_increases_thrust(self):
        ctrl = GeometricController()
        state = np.zeros(12)
        wrench = ctrl.compute(state, target_pos=np.array([0.0, 0.0, 1.0]))
        hover = ctrl.config.mass * ctrl.config.gravity
        assert wrench[0] > hover

    def test_returns_four_element_wrench(self):
        ctrl = GeometricController()
        wrench = ctrl.compute(np.zeros(12), np.zeros(3))
        assert wrench.shape == (4,)


class TestGeometricClosedLoop:
    def test_hover_convergence(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.5]))
        hover_force = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_force))

        ctrl = GeometricController()
        target = np.array([0.0, 0.0, 1.0])

        dt = 0.001
        for _ in range(5000):
            wrench = ctrl.compute(quad.state, target)
            quad.step(wrench, dt)

        assert abs(quad.position[2] - 1.0) < 0.15
