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

    Adds optional Gaussian noise to NDC output to simulate real-world
    detector jitter.

    Parameters
    ----------
    target_radius : effective radius of the target [m].
    ndc_noise_std : standard deviation of Gaussian noise on NDC (0 = perfect).
    seed : RNG seed for reproducibility.
    """

    def __init__(
        self,
        target_radius: float = 0.5,
        ndc_noise_std: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.target_radius = target_radius
        self.ndc_noise_std = ndc_noise_std
        self._rng = np.random.default_rng(seed)

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
        if visible and self.ndc_noise_std > 0:
            ndc = ndc + self._rng.normal(0, self.ndc_noise_std, size=2)
        return Detection(ndc, size_ratio, visible)


@dataclass
class VisualServoConfig:
    """Gains for image-based visual servoing."""

    kp_lateral: float = 1.5
    kp_forward: float = 1.0
    desired_size_ratio: float = 0.25
    max_velocity: float = 2.0
    desired_center_x: float = 0.0
    desired_center_y: float = 0.0
    kp_pan: float = 1.5
    kp_tilt: float = 3.0
    desired_tilt: float = -0.5


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

    def compute_from_gimbal(
        self,
        detection: Detection,
        yaw: float,
        pan: float,
        tilt: float,
    ) -> NDArray[np.floating]:
        """Return desired world velocity when a gimbal owns the centring loop.

        With an actively-pointed camera the bounding box sits at the image
        centre whatever the drone does, so image-centre error carries no
        information about where to fly.  The *gimbal angles* carry it
        instead: a non-zero pan means the target has drifted off the nose,
        and a tilt steeper than ``desired_tilt`` means the drone is too
        high above it.  Range is still closed on apparent size.

        Parameters
        ----------
        detection : current bounding box observation.
        yaw : drone heading [rad].
        pan : gimbal pan relative to the airframe [rad].
        tilt : gimbal tilt, negative looking down [rad].

        Returns
        -------
        (3,) velocity command in world frame.
        """
        if not detection.visible:
            return np.zeros(3)

        size_err = self.cfg.desired_size_ratio - detection.size_ratio

        # Pan is measured counter-clockwise from the nose and body +y is
        # left, so a positive pan asks for positive body-y motion.
        vx_body = self.cfg.kp_forward * size_err
        vy_body = self.cfg.kp_pan * pan
        vz = self.cfg.kp_tilt * (tilt - self.cfg.desired_tilt)

        cy, sy = np.cos(yaw), np.sin(yaw)
        vel = np.array([vx_body * cy - vy_body * sy, vx_body * sy + vy_body * cy, vz])
        speed = float(np.linalg.norm(vel))
        if speed > self.cfg.max_velocity:
            vel *= self.cfg.max_velocity / speed
        return vel

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

        err_lateral = detection.center_ndc[0] - self.cfg.desired_center_x
        err_vertical = detection.center_ndc[1] - self.cfg.desired_center_y
        size_err = self.cfg.desired_size_ratio - detection.size_ratio

        # Image +x is right and +y is down; body +y is left and world +z
        # is up (FLU / ENU), so both image errors flip sign on the way in.
        # A target drifting right of centre has to be chased to the right,
        # i.e. towards negative body-y.
        vx_body = self.cfg.kp_forward * size_err
        vy_body = -self.cfg.kp_lateral * err_lateral
        vz = -self.cfg.kp_lateral * err_vertical

        cy, sy = np.cos(yaw), np.sin(yaw)
        vx_world = vx_body * cy - vy_body * sy
        vy_world = vx_body * sy + vy_body * cy

        vel = np.array([vx_world, vy_world, vz])
        speed = float(np.linalg.norm(vel))
        if speed > self.cfg.max_velocity:
            vel *= self.cfg.max_velocity / speed
        return vel
