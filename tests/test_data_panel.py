# Erwin Lejeune - 2026-02-15
"""Tests for the data_panel visualization helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from uav_sim.visualization.data_panel import (
    setup_attitude_panel,
    setup_error_panel,
    setup_estimation_panel,
    setup_position_panel,
    setup_thrust_panel,
    setup_velocity_panel,
    update_attitude_panel,
    update_error_panel,
    update_estimation_panel,
    update_position_panel,
    update_thrust_panel,
    update_velocity_panel,
)


class TestPositionPanel:
    def test_setup_returns_artists(self):
        _, ax = plt.subplots()
        artists = setup_position_panel(ax)
        assert "lines" in artists
        assert len(artists["lines"]) == 6
        plt.close()

    def test_update_with_setpoint(self):
        _, ax = plt.subplots()
        artists = setup_position_panel(ax)
        t = np.linspace(0, 1, 50)
        actual = np.column_stack([t, 0.5 * t, t * 0])
        setpoint = np.ones_like(actual)
        update_position_panel(artists, t, actual, setpoint)
        assert ax.get_xlim()[1] >= 1.0
        plt.close()


class TestAttitudePanel:
    def test_update(self):
        _, ax = plt.subplots()
        artists = setup_attitude_panel(ax)
        t = np.linspace(0, 1, 30)
        euler = np.column_stack([0.1 * np.sin(t), np.zeros(30), np.zeros(30)])
        update_attitude_panel(artists, t, euler)
        plt.close()


class TestErrorPanel:
    def test_update(self):
        _, ax = plt.subplots()
        artists = setup_error_panel(ax)
        t = np.linspace(0, 1, 20)
        err = np.exp(-t)
        update_error_panel(artists, t, err)
        plt.close()


class TestEstimationPanel:
    def test_update_with_sigma(self):
        _, ax = plt.subplots()
        artists = setup_estimation_panel(ax)
        t = np.linspace(0, 2, 40)
        true_val = np.sin(t)
        est_val = true_val + 0.1 * np.random.randn(40)
        sigma = 0.2 * np.ones(40)
        update_estimation_panel(artists, t, true_val, est_val, sigma)
        plt.close()


class TestVelocityPanel:
    def test_update(self):
        _, ax = plt.subplots()
        artists = setup_velocity_panel(ax)
        t = np.linspace(0, 1, 25)
        vel = np.column_stack([np.ones(25), np.zeros(25), -0.5 * np.ones(25)])
        update_velocity_panel(artists, t, vel)
        plt.close()


class TestThrustPanel:
    def test_update_with_hover(self):
        _, ax = plt.subplots()
        artists = setup_thrust_panel(ax)
        t = np.linspace(0, 1, 20)
        thrust = 14.715 * np.ones(20)
        update_thrust_panel(artists, t, thrust, hover_thrust=14.715)
        plt.close()
