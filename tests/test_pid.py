# Erwin Lejeune - 2026-02-16
"""Tests for the cascaded PID controller."""

import numpy as np
import pytest

from uav_sim.path_tracking.pid_controller import (
    CascadedPIDController,
    PIDAxis,
    PIDGains,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class TestPIDAxis:
    def test_proportional_only(self):
        pid = PIDAxis(PIDGains(kp=2.0, ki=0.0, kd=0.0))
        out = pid.compute(1.0, dt=0.01)
        assert pytest.approx(out, rel=1e-6) == 2.0

    def test_integral_accumulates(self):
        pid = PIDAxis(PIDGains(kp=0.0, ki=1.0, kd=0.0))
        for _ in range(10):
            out = pid.compute(1.0, dt=0.1)
        assert pytest.approx(out, rel=1e-6) == 1.0  # 10 * 0.1 * 1.0

    def test_integral_anti_windup(self):
        pid = PIDAxis(PIDGains(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.5))
        for _ in range(100):
            pid.compute(1.0, dt=0.1)
        assert pid.integral == 0.5

    def test_derivative(self):
        pid = PIDAxis(PIDGains(kp=0.0, ki=0.0, kd=1.0))
        pid.compute(0.0, dt=0.01)  # first step (prev_error = 0)
        out = pid.compute(1.0, dt=0.01)  # error jumped from 0 to 1
        assert pytest.approx(out, rel=1e-6) == 100.0  # 1.0/0.01

    def test_reset(self):
        pid = PIDAxis(PIDGains(kp=1.0, ki=1.0, kd=0.0))
        pid.compute(5.0, dt=0.1)
        pid.reset()
        assert pid.integral == 0.0
        assert pid.prev_error == 0.0


class TestCascadedPIDController:
    def test_hover_thrust(self):
        """At target position with zero state, thrust ≈ m*g."""
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        target = np.array([0.0, 0.0, 0.0])
        wrench = ctrl.compute(state, target, dt=0.001)
        expected_thrust = ctrl.config.mass * ctrl.config.gravity
        assert pytest.approx(wrench[0], rel=0.1) == expected_thrust

    def test_zero_torque_at_hover(self):
        """When at target position and zero angles, torques should be zero."""
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        target = np.array([0.0, 0.0, 0.0])
        wrench = ctrl.compute(state, target, dt=0.001)
        np.testing.assert_allclose(wrench[1:], 0.0, atol=1e-6)

    def test_position_error_increases_thrust(self):
        """Quadrotor below target → more thrust."""
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        state[2] = -1.0  # 1m below target
        target = np.array([0.0, 0.0, 0.0])
        wrench = ctrl.compute(state, target, dt=0.001)
        hover_thrust = ctrl.config.mass * ctrl.config.gravity
        assert wrench[0] > hover_thrust

    def test_closed_loop_hover(self):
        """PID + quadrotor should converge to hover at target altitude."""
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.5]))
        hover_force = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_force))

        ctrl = CascadedPIDController()
        target = np.array([0.0, 0.0, 1.0])

        dt = 0.001
        for _ in range(5000):
            wrench = ctrl.compute(quad.state, target, dt=dt)
            quad.step(wrench, dt)

        assert abs(quad.position[2] - 1.0) < 0.1

    def test_closed_loop_xy_tracking(self):
        """PID should drive quadrotor towards a small XY offset."""
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 1.0]))
        hover_force = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_force))

        ctrl = CascadedPIDController()
        target = np.array([0.1, 0.0, 1.0])

        dt = 0.001
        for _ in range(3000):
            wrench = ctrl.compute(quad.state, target, dt=dt)
            quad.step(wrench, dt)

        # Quadrotor should be moving towards the X target.
        assert quad.position[0] > 0.01

    def test_reset_clears_integrators(self):
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        state[2] = -1.0
        ctrl.compute(state, np.zeros(3), dt=0.01)
        ctrl.reset()
        assert ctrl.pid_z.integral == 0.0

    def test_max_tilt_clamps_extreme_targets(self):
        """Large position error should not produce roll/pitch beyond max_tilt."""
        ctrl = CascadedPIDController()
        state = np.zeros(12)
        state[2] = 1.0
        # Target 100m away → would produce extreme angle without clamping
        target = np.array([100.0, 100.0, 1.0])
        wrench = ctrl.compute(state, target, dt=0.01)
        # Thrust should be capped (not blow up)
        hover_T = ctrl.config.mass * ctrl.config.gravity
        assert wrench[0] <= hover_T * ctrl.config.max_thrust_ratio + 1e-6
        assert wrench[0] >= 0.0

    def test_large_error_stable_closed_loop(self):
        """With tilt clamping, a large target should not cause NaN or flip."""
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 1.0]))
        ctrl = CascadedPIDController()
        target = np.array([10.0, 10.0, 1.0])

        dt = 0.005
        for _ in range(2000):
            wrench = ctrl.compute(quad.state, target, dt=dt)
            quad.step(wrench, dt)

        assert not np.any(np.isnan(quad.state))
        assert np.all(np.abs(quad.state[:3]) < 500)
