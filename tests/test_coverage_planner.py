# Erwin Lejeune - 2026-02-18
"""Tests for coverage path planner."""

import numpy as np
import pytest

from uav_sim.path_planning.coverage_planner import CoveragePathPlanner, CoverageRegion


class TestCoveragePathPlanner:
    def _region(self) -> CoverageRegion:
        return CoverageRegion(origin=np.array([0.0, 0.0]), width=20.0, height=20.0, altitude=10.0)

    def test_plan_returns_3d_waypoints(self):
        planner = CoveragePathPlanner(swath_width=5.0, overlap=0.1)
        path = planner.plan(self._region())
        assert path.ndim == 2
        assert path.shape[1] == 3
        assert len(path) > 0

    def test_altitude_is_constant(self):
        planner = CoveragePathPlanner(swath_width=5.0)
        path = planner.plan(self._region())
        np.testing.assert_allclose(path[:, 2], 10.0)

    def test_points_within_region(self):
        planner = CoveragePathPlanner(swath_width=4.0, margin=1.0)
        region = self._region()
        path = planner.plan(region)
        assert np.all(path[:, 0] >= region.origin[0])
        assert np.all(path[:, 0] <= region.origin[0] + region.width)
        assert np.all(path[:, 1] >= region.origin[1])
        assert np.all(path[:, 1] <= region.origin[1] + region.height)

    def test_estimated_coverage(self):
        planner = CoveragePathPlanner(swath_width=5.0, overlap=0.0)
        cov = planner.estimated_coverage(self._region())
        assert 0.5 < cov <= 1.0

    def test_swath_from_camera(self):
        swath = CoveragePathPlanner.swath_from_camera(altitude=10.0, h_fov=np.radians(60))
        expected = 2 * 10.0 * np.tan(np.radians(30))
        np.testing.assert_allclose(swath, expected, rtol=1e-6)

    def test_overlap_validation(self):
        with pytest.raises(ValueError):
            CoveragePathPlanner(overlap=1.0)
