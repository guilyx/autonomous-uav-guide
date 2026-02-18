# Erwin Lejeune - 2026-02-16
"""Tests for the 6DOF Quadrotor dynamics model."""

import numpy as np
import pytest

from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


@pytest.fixture
def quad() -> Quadrotor:
    return Quadrotor()


class TestQuadrotorInit:
    def test_default_state_is_zeros(self, quad: Quadrotor):
        np.testing.assert_array_equal(quad.state, np.zeros(12))

    def test_state_size(self, quad: Quadrotor):
        assert quad.state.shape == (12,)

    def test_time_starts_at_zero(self, quad: Quadrotor):
        assert quad.time == 0.0


class TestQuadrotorReset:
    def test_reset_position(self, quad: Quadrotor):
        pos = np.array([1.0, 2.0, 3.0])
        quad.reset(position=pos)
        np.testing.assert_array_equal(quad.position, pos)

    def test_reset_clears_velocity(self, quad: Quadrotor):
        quad.state[6:9] = [1.0, 2.0, 3.0]
        quad.reset()
        np.testing.assert_array_equal(quad.velocity, [0.0, 0.0, 0.0])


class TestRotationMatrix:
    def test_identity_at_zero_angles(self):
        R = Quadrotor.rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_is_orthogonal(self):
        R = Quadrotor.rotation_matrix(0.3, 0.2, 0.5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_determinant_is_one(self):
        R = Quadrotor.rotation_matrix(0.7, -0.4, 1.2)
        assert pytest.approx(np.linalg.det(R), abs=1e-12) == 1.0


class TestHover:
    def test_hover_wrench_magnitude(self, quad: Quadrotor):
        wrench = quad.hover_wrench()
        expected_thrust = quad.params.mass * quad.params.gravity
        assert pytest.approx(wrench[0], rel=1e-6) == expected_thrust
        np.testing.assert_array_equal(wrench[1:], [0.0, 0.0, 0.0])

    def test_hover_maintains_altitude(self, quad: Quadrotor):
        """Quadrotor should stay near z=1.0 when motors are pre-spun to hover."""
        quad.reset(position=np.array([0.0, 0.0, 1.0]))
        wrench = quad.hover_wrench()
        # Pre-spin motors to hover speed so there is no transient drop.
        hover_force = wrench[0] / 4.0
        for motor in quad.motors:
            motor.reset(motor.thrust_to_omega(hover_force))
        for _ in range(1000):
            quad.step(wrench, dt=0.001)
        assert abs(quad.position[2] - 1.0) < 0.05

    def test_hover_maintains_zero_rotation(self, quad: Quadrotor):
        """Euler angles should stay near zero during hover."""
        quad.reset(position=np.array([0.0, 0.0, 1.0]))
        wrench = quad.hover_wrench()
        for _ in range(500):
            quad.step(wrench, dt=0.001)
        np.testing.assert_allclose(quad.euler, [0.0, 0.0, 0.0], atol=0.01)


class TestFreefall:
    def test_freefall_acceleration(self, quad: Quadrotor):
        """With zero wrench, quad should accelerate downward at g."""
        quad.reset(position=np.array([0.0, 0.0, 10.0]))
        zero_wrench = np.array([0.0, 0.0, 0.0, 0.0])
        for _ in range(100):
            quad.step(zero_wrench, dt=0.001)
        # After 0.1s of freefall, vz ≈ -g * t ≈ -0.981
        assert quad.velocity[2] < -0.9


class TestMotorIntegration:
    def test_motor_speeds_after_hover(self, quad: Quadrotor):
        """Motors should converge to equal speeds for hover."""
        quad.reset()
        wrench = quad.hover_wrench()
        for _ in range(500):
            quad.step(wrench, dt=0.001)
        speeds = quad.get_motor_speeds()
        assert np.std(speeds) < 1.0  # all motors nearly equal
        assert np.all(speeds > 0.0)


class TestTimekeeping:
    def test_time_advances(self, quad: Quadrotor):
        wrench = quad.hover_wrench()
        for _ in range(100):
            quad.step(wrench, dt=0.01)
        assert pytest.approx(quad.time, abs=1e-6) == 1.0
