# Erwin Lejeune - 2026-02-16
"""Tests for the control allocation Mixer."""

import numpy as np
import pytest

from uav_sim.vehicles.components.mixer import Mixer


@pytest.fixture
def x_mixer() -> Mixer:
    return Mixer(arm_length=0.175, frame="x")


@pytest.fixture
def plus_mixer() -> Mixer:
    return Mixer(arm_length=0.175, frame="+")


class TestMixerInit:
    def test_x_frame_matrix_shape(self, x_mixer: Mixer):
        assert x_mixer.mix_matrix.shape == (4, 4)

    def test_plus_frame_matrix_shape(self, plus_mixer: Mixer):
        assert plus_mixer.mix_matrix.shape == (4, 4)

    def test_unknown_frame_raises(self):
        with pytest.raises(ValueError, match="Unknown frame"):
            Mixer(frame="h")


class TestWrenchToForces:
    def test_hover_produces_equal_forces(self, x_mixer: Mixer):
        hover_thrust = 1.5 * 9.81  # ~14.715 N
        wrench = np.array([hover_thrust, 0.0, 0.0, 0.0])
        forces = x_mixer.wrench_to_forces(wrench)
        expected = hover_thrust / 4.0
        np.testing.assert_allclose(forces, expected, atol=1e-8)

    def test_forces_are_non_negative(self, x_mixer: Mixer):
        wrench = np.array([0.0, 0.0, 0.0, 0.0])
        forces = x_mixer.wrench_to_forces(wrench)
        assert np.all(forces >= 0.0)


class TestRoundTrip:
    def test_wrench_round_trip_no_clamping(self, x_mixer: Mixer):
        """When all resulting forces are positive, wrench should round-trip exactly."""
        wrench = np.array([1.0, 0.001, 0.001, 0.0001])
        forces = x_mixer.wrench_to_forces(wrench)
        assert np.all(forces > 0.0), "Test assumes no clamping"
        recovered = x_mixer.forces_to_wrench(forces)
        np.testing.assert_allclose(recovered, wrench, atol=1e-10)

    def test_forces_round_trip(self, x_mixer: Mixer):
        forces = np.array([0.3, 0.2, 0.3, 0.2])
        wrench = x_mixer.forces_to_wrench(forces)
        recovered = x_mixer.wrench_to_forces(wrench)
        np.testing.assert_allclose(recovered, forces, atol=1e-10)


class TestPlusFrame:
    def test_hover_produces_equal_forces(self, plus_mixer: Mixer):
        wrench = np.array([1.0, 0.0, 0.0, 0.0])
        forces = plus_mixer.wrench_to_forces(wrench)
        np.testing.assert_allclose(forces, 0.25, atol=1e-8)

    def test_round_trip(self, plus_mixer: Mixer):
        wrench = np.array([1.0, 0.005, 0.005, 0.0005])
        forces = plus_mixer.wrench_to_forces(wrench)
        recovered = plus_mixer.forces_to_wrench(forces)
        np.testing.assert_allclose(recovered, wrench, atol=1e-10)
