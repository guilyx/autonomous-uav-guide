# Erwin Lejeune - 2026-02-18
"""Tests for environment presets."""

import numpy as np

from uav_sim.environment import (
    EnvironmentPreset,
    city_world,
    create_environment,
    default_world,
    indoor_world,
    open_field,
)


class TestDefaultWorld:
    def test_returns_world_and_buildings(self):
        w, b = default_world()
        assert w is not None
        assert len(b) > 0

    def test_size_30(self):
        w, _ = default_world()
        np.testing.assert_allclose(w.bounds_max, 30.0)


class TestCityWorld:
    def test_larger_than_default(self):
        w, b = city_world()
        assert w.bounds_max[0] == 50.0
        assert len(b) >= 6


class TestIndoorWorld:
    def test_low_ceiling(self):
        w, obs = indoor_world()
        assert w.bounds_max[2] == 3.0
        assert len(obs) >= 2


class TestOpenField:
    def test_no_obstacles(self):
        w, b = open_field()
        assert len(b) == 0
        assert w.bounds_max[0] == 60.0


class TestCreateEnvironment:
    def test_city(self):
        w, b = create_environment(EnvironmentPreset.CITY)
        assert w.bounds_max[0] == 50.0

    def test_indoor(self):
        w, _ = create_environment(EnvironmentPreset.INDOOR)
        assert w.bounds_max[2] == 3.0

    def test_open_field(self):
        _, b = create_environment(EnvironmentPreset.OPEN_FIELD)
        assert len(b) == 0
