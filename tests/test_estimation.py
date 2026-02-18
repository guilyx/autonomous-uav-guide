# Erwin Lejeune - 2026-02-16
"""Tests for state estimation algorithms."""

import numpy as np
import pytest

from uav_sim.estimation.complementary_filter import ComplementaryFilter
from uav_sim.estimation.ekf import ExtendedKalmanFilter
from uav_sim.estimation.particle_filter import ParticleFilter
from uav_sim.estimation.ukf import UnscentedKalmanFilter

# ---------------------------------------------------------------------------
# Complementary filter
# ---------------------------------------------------------------------------


class TestComplementaryFilter:
    def test_initial_attitude_is_zero(self):
        cf = ComplementaryFilter()
        np.testing.assert_array_equal(cf.attitude, [0.0, 0.0])

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            ComplementaryFilter(alpha=1.5)

    def test_pure_accel(self):
        """With alpha=0 the filter should follow the accelerometer exactly."""
        cf = ComplementaryFilter(alpha=0.01)
        accel = np.array([0.0, 0.0, 9.81])  # level
        gyro = np.array([0.0, 0.0, 0.0])
        for _ in range(100):
            cf.update(gyro, accel, dt=0.01)
        np.testing.assert_allclose(cf.attitude, [0.0, 0.0], atol=0.05)

    def test_detects_tilt(self):
        """Tilted accel should produce nonzero pitch."""
        cf = ComplementaryFilter(alpha=0.5)
        accel = np.array([2.0, 0.0, 9.81])  # tilted forward
        gyro = np.array([0.0, 0.0, 0.0])
        for _ in range(50):
            cf.update(gyro, accel, dt=0.01)
        assert abs(cf.pitch) > 0.1

    def test_reset(self):
        cf = ComplementaryFilter()
        cf.update(np.zeros(3), np.array([0.0, 0.0, 9.81]), dt=0.01)
        cf.reset(0.5, 0.5)
        assert cf.roll == 0.5
        assert cf.pitch == 0.5


# ---------------------------------------------------------------------------
# EKF tests (with a simple linear system for validation)
# ---------------------------------------------------------------------------


def _linear_f(x, u, dt):
    """Simple 1D constant velocity: x = [pos, vel]."""
    return np.array([x[0] + x[1] * dt, x[1]])


def _linear_h(x):
    """Measure position only."""
    return np.array([x[0]])


def _linear_F_jac(x, u, dt):
    return np.array([[1.0, dt], [0.0, 1.0]])


def _linear_H_jac(x):
    return np.array([[1.0, 0.0]])


class TestEKF:
    def test_predict_advances_state(self):
        ekf = ExtendedKalmanFilter(2, 1, _linear_f, _linear_h, _linear_F_jac, _linear_H_jac)
        ekf.reset(np.array([0.0, 1.0]))  # pos=0, vel=1
        ekf.Q = np.eye(2) * 0.001
        ekf.predict(np.zeros(1), dt=1.0)
        assert pytest.approx(ekf.state[0], abs=0.1) == 1.0  # moved by vel*dt

    def test_update_corrects_state(self):
        ekf = ExtendedKalmanFilter(2, 1, _linear_f, _linear_h, _linear_F_jac, _linear_H_jac)
        ekf.reset(np.array([0.0, 0.0]))
        ekf.R = np.eye(1) * 0.01
        ekf.predict(np.zeros(1), dt=1.0)
        ekf.update(np.array([5.0]))  # observe pos=5
        assert ekf.state[0] > 1.0  # pulled towards measurement

    def test_covariance_shrinks_after_update(self):
        ekf = ExtendedKalmanFilter(2, 1, _linear_f, _linear_h, _linear_F_jac, _linear_H_jac)
        ekf.reset(np.array([0.0, 0.0]))
        ekf.predict(np.zeros(1), dt=1.0)
        P_before = ekf.covariance[0, 0]
        ekf.update(np.array([0.0]))
        P_after = ekf.covariance[0, 0]
        assert P_after < P_before

    def test_tracks_constant_velocity(self):
        """EKF should converge to true state with noisy measurements."""
        ekf = ExtendedKalmanFilter(2, 1, _linear_f, _linear_h, _linear_F_jac, _linear_H_jac)
        ekf.reset(np.array([0.0, 0.0]))
        ekf.Q = np.eye(2) * 0.001
        ekf.R = np.eye(1) * 1.0

        rng = np.random.default_rng(42)
        true_state = np.array([0.0, 1.0])
        for _ in range(100):
            true_state = _linear_f(true_state, np.zeros(1), 0.1)
            ekf.predict(np.zeros(1), dt=0.1)
            z = np.array([true_state[0] + rng.normal(0, 1.0)])
            ekf.update(z)

        assert abs(ekf.state[0] - true_state[0]) < 2.0
        assert abs(ekf.state[1] - 1.0) < 0.5


# ---------------------------------------------------------------------------
# UKF tests (same linear system)
# ---------------------------------------------------------------------------


class TestUKF:
    def test_predict_advances_state(self):
        ukf = UnscentedKalmanFilter(2, 1, _linear_f, _linear_h)
        ukf.reset(np.array([0.0, 1.0]))
        ukf.Q = np.eye(2) * 0.001
        ukf.predict(np.zeros(1), dt=1.0)
        assert pytest.approx(ukf.state[0], abs=0.1) == 1.0

    def test_update_corrects_state(self):
        ukf = UnscentedKalmanFilter(2, 1, _linear_f, _linear_h)
        ukf.reset(np.array([0.0, 0.0]))
        ukf.R = np.eye(1) * 0.01
        ukf.predict(np.zeros(1), dt=1.0)
        ukf.update(np.array([5.0]))
        assert ukf.state[0] > 1.0

    def test_tracks_constant_velocity(self):
        ukf = UnscentedKalmanFilter(2, 1, _linear_f, _linear_h)
        ukf.reset(np.array([0.0, 0.0]))
        ukf.Q = np.eye(2) * 0.001
        ukf.R = np.eye(1) * 1.0

        rng = np.random.default_rng(42)
        true_state = np.array([0.0, 1.0])
        for _ in range(100):
            true_state = _linear_f(true_state, np.zeros(1), 0.1)
            ukf.predict(np.zeros(1), dt=0.1)
            z = np.array([true_state[0] + rng.normal(0, 1.0)])
            ukf.update(z)

        assert abs(ukf.state[0] - true_state[0]) < 2.0


# ---------------------------------------------------------------------------
# Particle filter tests
# ---------------------------------------------------------------------------


def _pf_likelihood(z, x):
    """Gaussian likelihood p(z|x) for position measurement."""
    diff = z[0] - x[0]
    return float(np.exp(-0.5 * diff**2))


class TestParticleFilter:
    def test_estimate_near_init(self):
        pf = ParticleFilter(2, 500, _linear_f, _pf_likelihood, process_noise_std=0.01)
        pf.reset(np.array([0.0, 1.0]), spread=0.01)
        est = pf.estimate
        np.testing.assert_allclose(est, [0.0, 1.0], atol=0.1)

    def test_predict_advances(self):
        pf = ParticleFilter(2, 500, _linear_f, _pf_likelihood, process_noise_std=0.01)
        pf.reset(np.array([0.0, 1.0]), spread=0.01)
        pf.predict(np.zeros(1), dt=1.0)
        assert pf.estimate[0] > 0.5

    def test_update_corrects(self):
        pf = ParticleFilter(2, 1000, _linear_f, _pf_likelihood, process_noise_std=0.01)
        pf.reset(np.array([0.0, 0.0]), spread=1.0)
        pf.update(np.array([5.0]))
        assert pf.estimate[0] > 1.0

    def test_variance_decreases_after_update(self):
        pf = ParticleFilter(2, 1000, _linear_f, _pf_likelihood, process_noise_std=0.01)
        pf.reset(np.array([0.0, 0.0]), spread=2.0)
        var_before = pf.variance[0]
        pf.update(np.array([0.0]))
        var_after = pf.variance[0]
        assert var_after < var_before
