# Erwin Lejeune - 2026-02-15
"""Tests for vehicle presets."""

import numpy as np
import pytest

from uav_sim.vehicles.presets import VehiclePreset, create_quadrotor, get_params


class TestCreateQuadrotor:
    @pytest.mark.parametrize(
        "preset",
        [
            VehiclePreset.CRAZYFLIE,
            VehiclePreset.DJI_MINI,
            VehiclePreset.RACING_250,
            VehiclePreset.DJI_MATRICE,
        ],
    )
    def test_all_presets_instantiate(self, preset: VehiclePreset):
        quad = create_quadrotor(preset)
        assert quad.state.shape == (12,)
        assert quad.params.mass > 0

    def test_crazyflie_is_lightweight(self):
        quad = create_quadrotor(VehiclePreset.CRAZYFLIE)
        assert quad.params.mass < 0.1

    def test_matrice_is_heavy(self):
        quad = create_quadrotor(VehiclePreset.DJI_MATRICE)
        assert quad.params.mass > 3.0

    def test_override_mass(self):
        quad = create_quadrotor(VehiclePreset.RACING_250, mass=2.0)
        assert quad.params.mass == 2.0
        assert quad.params.arm_length == 0.175  # unchanged

    def test_custom_preset(self):
        quad = create_quadrotor(VehiclePreset.CUSTOM, mass=0.5, arm_length=0.1)
        assert quad.params.mass == 0.5

    def test_hover_wrench_scales_with_mass(self):
        light = create_quadrotor(VehiclePreset.CRAZYFLIE)
        heavy = create_quadrotor(VehiclePreset.DJI_MATRICE)
        assert heavy.hover_wrench()[0] > light.hover_wrench()[0]


class TestGetParams:
    def test_returns_correct_mass(self):
        p = get_params(VehiclePreset.DJI_MINI)
        assert p.mass == pytest.approx(0.249)

    def test_inertia_is_diagonal(self):
        p = get_params(VehiclePreset.RACING_250)
        off_diag = p.inertia - np.diag(np.diag(p.inertia))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)
