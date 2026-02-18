# Erwin Lejeune - 2026-02-18
"""Tests for gimbal controllers."""

import numpy as np

from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import (
    BBoxTracker,
    PointTracker,
    project_to_image,
)


class TestPointTracker:
    def test_tracks_point_below(self):
        g = Gimbal()
        pt = PointTracker(g)
        cam = np.array([5.0, 5.0, 10.0])
        target = np.array([5.0, 5.0, 0.0])
        for _ in range(100):
            pt.step(cam, target, yaw=0.0, dt=0.01)
        assert g.tilt < -0.5

    def test_tracks_point_forward(self):
        g = Gimbal()
        pt = PointTracker(g)
        cam = np.array([5.0, 5.0, 10.0])
        target = np.array([15.0, 5.0, 10.0])
        for _ in range(100):
            pt.step(cam, target, yaw=0.0, dt=0.01)
        assert abs(g.tilt) < 0.3


class TestBBoxTracker:
    def test_centres_bbox(self):
        g = Gimbal()
        bt = BBoxTracker(g)
        initial_pan = g.pan
        bt.step(np.array([0.5, 0.0]), 0.3, dt=0.05)
        assert g.pan != initial_pan


class TestProjectToImage:
    def test_point_in_front(self):
        g = Gimbal()
        cam = np.array([0.0, 0.0, 10.0])
        target = np.array([0.0, 0.0, 0.0])
        g.step(*g.look_at(cam, target, 0.0), dt=1.0)
        ndc, vis = project_to_image(target, cam, g, h_fov=1.0, v_fov=0.8, yaw=0.0)
        assert vis or abs(ndc[0]) <= 2.0

    def test_point_behind(self):
        g = Gimbal()
        cam = np.array([0.0, 0.0, 10.0])
        target = np.array([0.0, 0.0, 20.0])
        ndc, vis = project_to_image(target, cam, g, h_fov=1.0, v_fov=0.8, yaw=0.0)
        assert not vis
