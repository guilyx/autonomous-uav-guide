# Erwin Lejeune - 2026-02-16
"""Tests for the Motor model."""

import pytest

from uav_sim.vehicles.components.motor import Motor


@pytest.fixture
def motor() -> Motor:
    return Motor(k_thrust=2.98e-6, k_torque=1.14e-7, tau=0.02, omega_max=2199.0)


class TestMotorInit:
    def test_initial_omega_is_zero(self, motor: Motor):
        assert motor.omega == 0.0

    def test_initial_thrust_is_zero(self, motor: Motor):
        assert motor.thrust == 0.0

    def test_initial_torque_is_zero(self, motor: Motor):
        assert motor.torque == 0.0


class TestMotorReset:
    def test_reset_to_value(self, motor: Motor):
        motor.reset(1000.0)
        assert motor.omega == 1000.0

    def test_reset_clamps_to_max(self, motor: Motor):
        motor.reset(99999.0)
        assert motor.omega == motor.omega_max


class TestMotorStep:
    def test_step_converges_towards_command(self, motor: Motor):
        cmd = 1500.0
        for _ in range(500):
            motor.step(cmd, dt=0.001)
        assert abs(motor.omega - cmd) < 1.0

    def test_step_clamps_command(self, motor: Motor):
        motor.step(99999.0, dt=1.0)
        assert motor.omega <= motor.omega_max

    def test_first_order_lag_behaviour(self, motor: Motor):
        motor.reset(0.0)
        motor.step(1000.0, dt=0.02)
        # After one time constant, should be ~50% of command
        assert 400.0 < motor.omega < 600.0


class TestMotorThrustCurve:
    def test_thrust_is_quadratic(self, motor: Motor):
        motor.reset(100.0)
        t1 = motor.thrust
        motor.reset(200.0)
        t2 = motor.thrust
        assert pytest.approx(t2, rel=1e-6) == 4.0 * t1

    def test_torque_sign_ccw(self):
        m = Motor(direction=1)
        m.reset(100.0)
        assert m.torque > 0.0

    def test_torque_sign_cw(self):
        m = Motor(direction=-1)
        m.reset(100.0)
        assert m.torque < 0.0


class TestThrustToOmega:
    def test_round_trip(self, motor: Motor):
        motor.reset(1200.0)
        thrust = motor.thrust
        omega_recovered = motor.thrust_to_omega(thrust)
        assert pytest.approx(omega_recovered, rel=1e-6) == 1200.0

    def test_negative_thrust_returns_zero(self, motor: Motor):
        assert motor.thrust_to_omega(-10.0) == 0.0


class TestGetState:
    def test_state_shape(self, motor: Motor):
        state = motor.get_state()
        assert state.shape == (3,)

    def test_state_values(self, motor: Motor):
        motor.reset(500.0)
        state = motor.get_state()
        assert state[0] == motor.omega
        assert state[1] == motor.thrust
        assert state[2] == motor.torque
