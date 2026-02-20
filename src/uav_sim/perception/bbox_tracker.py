# Erwin Lejeune - 2026-02-18
"""Simulated bounding-box object detector + visual servoing controller.

Provides:
- ``SimulatedDetector``: projects known 3D targets into the camera to
  produce fake 2D bounding boxes (no ML required).
- ``VisualServoController``: generates velocity commands to keep the
  target centred and at a desired distance using image-based feedback.

Together they enable a "follow the bounding box" demo where the drone
autonomously tracks a moving or static ground target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import project_to_image


@dataclass
class Detection:
    """A single 2D bounding box detection."""

    center_ndc: NDArray[np.floating]  # (2,) normalised image coords [-1, 1]
    size_ratio: float  # bbox diagonal / image diagonal
    visible: bool


class SimulatedDetector:
    """Projects a known 3D target into the camera image as a bounding box.

    The "detection" is perfect (no noise, no ML). The target is modelled
    as a sphere of given radius; its projected size determines bbox_size.

    Parameters
    ----------
    target_radius : effective radius of the target [m].
    """

    def __init__(self, target_radius: float = 0.5) -> None:
        self.target_radius = target_radius

    def detect(
        self,
        target_pos: NDArray[np.floating],
        camera_pos: NDArray[np.floating],
        gimbal: Gimbal,
        h_fov: float,
        v_fov: float,
        yaw: float = 0.0,
    ) -> Detection:
        """Return a Detection of the target from the current camera pose."""
        ndc, visible = project_to_image(target_pos, camera_pos, gimbal, h_fov, v_fov, yaw)
        dist = float(np.linalg.norm(target_pos - camera_pos))
        if dist < 0.01:
            return Detection(ndc, 1.0, False)
        angular_size = 2 * np.arctan(self.target_radius / dist)
        size_ratio = float(angular_size / max(h_fov, v_fov))
        return Detection(ndc, size_ratio, visible)


@dataclass
class VisualServoConfig:
    """Gains for image-based visual servoing."""

    kp_lateral: float = 1.5
    kp_forward: float = 1.0
    desired_size_ratio: float = 0.25
    max_velocity: float = 2.0


class VisualServoController:
    """Generate velocity commands to track a bounding box.

    Uses proportional control on:
    - lateral/vertical error: drives bbox to image centre
    - size error: drives forward/backward to maintain desired apparent size

    Parameters
    ----------
    config : gains and desired size.
    """

    def __init__(self, config: VisualServoConfig | None = None) -> None:
        self.cfg = config or VisualServoConfig()

    def compute(
        self,
        detection: Detection,
        yaw: float,
    ) -> NDArray[np.floating]:
        """Return desired world-frame velocity [vx, vy, vz].

        Parameters
        ----------
        detection : current bounding box observation.
        yaw : drone heading.

        Returns
        -------
        (3,) velocity command in world frame.
        """
        if not detection.visible:
            return np.zeros(3)

        err_lateral = detection.center_ndc[0]
        err_vertical = -detection.center_ndc[1]
        size_err = self.cfg.desired_size_ratio - detection.size_ratio

        vx_body = self.cfg.kp_forward * size_err
        vy_body = self.cfg.kp_lateral * err_lateral
        vz = self.cfg.kp_lateral * err_vertical

        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_world = vx_body * cy - vy_body * sy
        vy_world = vx_body * sy + vy_body * cy

        vel = np.array([vx_world, vy_world, vz])
        speed = float(np.linalg.norm(vel))
        if speed > self.cfg.max_velocity:
            vel *= self.cfg.max_velocity / speed
        return vel
