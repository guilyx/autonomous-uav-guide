# Erwin Lejeune - 2026-02-15
"""Tests for the layered control stack."""

import numpy as np
import pytest

from uav_sim.control import (
    AttitudeController,
    ControlMode,
    FlightController,
    PositionController,
    RateController,
    VelocityController,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class TestRateController:
    def test_zero_error_zero_output(self):
        rc = RateController()
        torques = rc.compute(np.zeros(3), np.zeros(3), dt=0.01)
        np.testing.assert_allclose(torques, 0.0, atol=1e-10)

    def test_positive_error_positive_output(self):
        rc = RateController()
        torques = rc.compute(np.zeros(3), np.array([1.0, 0.0, 0.0]), dt=0.01)
        assert torques[0] > 0.0

    def test_proportional_scaling(self):
        rc = RateController()
        t1 = rc.compute(np.zeros(3), np.array([1.0, 0.0, 0.0]), dt=0.01)
        t2 = rc.compute(np.zeros(3), np.array([2.0, 0.0, 0.0]), dt=0.01)
        np.testing.assert_allclose(t2[0], 2.0 * t1[0], rtol=1e-10)

    def test_conservative_torque_magnitude(self):
        """1 rad/s error should NOT produce > 0.5 Nm torque (avoid oscillation)."""
        rc = RateController()
        torques = rc.compute(np.zeros(3), np.array([1.0, 1.0, 1.0]), dt=0.01)
        assert all(abs(t) < 0.5 for t in torques)


class TestAttitudeController:
    def test_zero_error_gives_hover_thrust(self):
        ac = AttitudeController()
        wrench = ac.compute(
            euler=np.zeros(3),
            angular_velocity=np.zeros(3),
            desired_euler=np.zeros(3),
            thrust=14.715,
            dt=0.01,
        )
        assert wrench.shape == (4,)
        assert pytest.approx(wrench[0], abs=0.01) == 14.715

    def test_nonzero_roll_error(self):
        ac = AttitudeController()
        wrench = ac.compute(
            euler=np.zeros(3),
            angular_velocity=np.zeros(3),
            desired_euler=np.array([0.1, 0.0, 0.0]),
            thrust=14.715,
            dt=0.01,
        )
        assert wrench[1] > 0.0

    def test_zero_error_zero_torques(self):
        ac = AttitudeController()
        wrench = ac.compute(
            euler=np.array([0.1, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            desired_euler=np.array([0.1, 0.0, 0.0]),
            thrust=14.715,
            dt=0.01,
        )
        np.testing.assert_allclose(wrench[1:], 0.0, atol=1e-10)


class TestVelocityController:
    def test_at_target_gives_hover(self):
        vc = VelocityController()
        des_euler, thrust = vc.compute(
            velocity=np.zeros(3),
            yaw=0.0,
            desired_velocity=np.zeros(3),
            dt=0.01,
        )
        assert pytest.approx(thrust, rel=0.01) == 1.5 * 9.81
        np.testing.assert_allclose(des_euler[:2], 0.0, atol=0.01)

    def test_forward_velocity_tilts(self):
        vc = VelocityController()
        des_euler, thrust = vc.compute(
            velocity=np.zeros(3),
            yaw=0.0,
            desired_velocity=np.array([2.0, 0.0, 0.0]),
            dt=0.01,
        )
        assert abs(des_euler[1]) > 0.01


class TestPositionController:
    def test_at_target_zero_velocity(self):
        pc = PositionController()
        vel = pc.compute(
            np.array([5.0, 5.0, 10.0]),
            np.array([5.0, 5.0, 10.0]),
            dt=0.01,
        )
        np.testing.assert_allclose(vel, 0.0, atol=1e-6)

    def test_velocity_is_clamped(self):
        pc = PositionController()
        vel = pc.compute(np.zeros(3), np.array([100.0, 100.0, 100.0]), dt=0.01)
        assert float(np.linalg.norm(vel)) <= pc.cfg.max_velocity + 1e-6

    def test_velocity_feedback_damps(self):
        """Providing measured velocity should reduce commanded velocity."""
        pc = PositionController()
        vel_no_fb = pc.compute(np.zeros(3), np.array([1.0, 0.0, 0.0]), dt=0.01)
        vel_fb = pc.compute(
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
            dt=0.01,
            velocity=np.array([0.5, 0.0, 0.0]),
        )
        assert float(np.linalg.norm(vel_fb)) < float(np.linalg.norm(vel_no_fb))


class TestFlightController:
    def test_position_mode_hover(self):
        fc = FlightController()
        state = np.zeros(12)
        state[2] = 5.0
        fc.set_position_target(np.array([0.0, 0.0, 5.0]))
        wrench = fc.compute(state, dt=0.005)
        assert wrench.shape == (4,)
        assert wrench[0] > 0.0

    def test_velocity_mode(self):
        fc = FlightController()
        fc.set_velocity_target(np.array([1.0, 0.0, 0.0]))
        assert fc.mode == ControlMode.VELOCITY
        wrench = fc.compute(np.zeros(12), dt=0.005)
        assert wrench.shape == (4,)

    def test_closed_loop_hover(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 5.0]))
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))

        fc = FlightController()
        fc.set_position_target(np.array([0.0, 0.0, 5.0]))

        dt = 0.005
        for _ in range(2000):
            wrench = fc.compute(quad.state, dt)
            quad.step(wrench, dt)

        assert abs(quad.position[2] - 5.0) < 0.3

    def test_closed_loop_position_tracking(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 5.0]))
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))

        fc = FlightController()
        fc.set_position_target(np.array([3.0, 0.0, 5.0]))

        dt = 0.005
        for _ in range(6000):
            wrench = fc.compute(quad.state, dt)
            quad.step(wrench, dt)

        assert abs(quad.position[0] - 3.0) < 1.5
        assert abs(quad.position[2] - 5.0) < 2.0

    def test_reset(self):
        fc = FlightController()
        fc.set_position_target(np.array([10.0, 10.0, 10.0]))
        fc.reset()
        np.testing.assert_array_equal(fc._target_pos, np.zeros(3))
