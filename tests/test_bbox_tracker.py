# Erwin Lejeune - 2026-02-18
"""Tests for bounding box tracker and visual servoing."""

import numpy as np

from uav_sim.perception.bbox_tracker import (
    Detection,
    SimulatedDetector,
    VisualServoController,
)
from uav_sim.sensors.gimbal import Gimbal


class TestSimulatedDetector:
    def test_visible_below(self):
        g = Gimbal()
        cam = np.array([5.0, 5.0, 10.0])
        target = np.array([5.0, 5.0, 0.0])
        g.step(*g.look_at(cam, target, 0.0), dt=1.0)
        det = SimulatedDetector(target_radius=0.5)
        d = det.detect(target, cam, g, h_fov=1.0, v_fov=0.8)
        assert d.size_ratio > 0

    def test_invisible_behind(self):
        g = Gimbal()
        cam = np.array([0.0, 0.0, 10.0])
        target = np.array([0.0, 0.0, 20.0])
        det = SimulatedDetector()
        d = det.detect(target, cam, g, h_fov=1.0, v_fov=0.8)
        assert not d.visible


class TestVisualServo:
    def test_centred_no_lateral(self):
        ctrl = VisualServoController()
        det = Detection(center_ndc=np.array([0.0, 0.0]), size_ratio=0.25, visible=True)
        vel = ctrl.compute(det, yaw=0.0)
        assert abs(vel[1]) < 0.01

    def test_invisible_zero_vel(self):
        ctrl = VisualServoController()
        det = Detection(center_ndc=np.array([0.5, 0.0]), size_ratio=0.1, visible=False)
        vel = ctrl.compute(det, yaw=0.0)
        np.testing.assert_allclose(vel, 0.0)

    def test_off_centre_corrects(self):
        ctrl = VisualServoController()
        det = Detection(center_ndc=np.array([0.5, 0.0]), size_ratio=0.25, visible=True)
        vel = ctrl.compute(det, yaw=0.0)
        assert float(np.linalg.norm(vel)) > 0.01
