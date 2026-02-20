# Erwin Lejeune - 2026-02-18
"""Gimbal pointing controllers.

Provides closed-loop controllers that command a :class:`Gimbal` to:
- Track a static/moving world-frame point (``PointTracker``)
- Keep a bounding box centred in the image (``BBoxTracker``)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.gimbal import Gimbal


@dataclass
class PointTrackerConfig:
    """Gains for the point-tracking gimbal controller."""

    kp_pan: float = 2.0
    kp_tilt: float = 2.0


class PointTracker:
    """Commands gimbal to look at a world-frame point.

    At each step, computes the desired angles via ``gimbal.look_at``
    then commands the gimbal with rate-limited steps.

    Parameters
    ----------
    gimbal : the gimbal to control.
    config : proportional gains (unused for now; look_at is geometric).
    """

    def __init__(
        self,
        gimbal: Gimbal,
        config: PointTrackerConfig | None = None,
    ) -> None:
        self.gimbal = gimbal
        self.cfg = config or PointTrackerConfig()

    def step(
        self,
        camera_pos: NDArray[np.floating],
        target: NDArray[np.floating],
        yaw: float,
        dt: float,
    ) -> None:
        """Command the gimbal to point at *target* using P-control on look-at error."""
        des_pan, des_tilt = self.gimbal.look_at(camera_pos, target, yaw)
        cmd_pan = self.gimbal.pan + self.cfg.kp_pan * (des_pan - self.gimbal.pan)
        cmd_tilt = self.gimbal.tilt + self.cfg.kp_tilt * (des_tilt - self.gimbal.tilt)
        self.gimbal.step(cmd_pan, cmd_tilt, dt)


@dataclass
class BBoxTrackerConfig:
    """Gains for keeping a bounding box centred in the image."""

    kp_pan: float = 1.5
    kp_tilt: float = 1.5
    desired_size_ratio: float = 0.3  # target bbox / image fraction


class BBoxTracker:
    """Commands gimbal to centre a bounding box in the camera image.

    The controller receives the bounding box centre in normalised image
    coordinates ([-1, 1] x [-1, 1], where (0,0) = image centre) and
    adjusts pan/tilt to null the error.

    Parameters
    ----------
    gimbal : the gimbal to control.
    config : proportional gains.
    """

    def __init__(
        self,
        gimbal: Gimbal,
        config: BBoxTrackerConfig | None = None,
    ) -> None:
        self.gimbal = gimbal
        self.cfg = config or BBoxTrackerConfig()

    def step(
        self,
        bbox_center_norm: NDArray[np.floating],
        bbox_size_ratio: float,
        dt: float,
    ) -> None:
        """Adjust gimbal to centre the bounding box.

        Parameters
        ----------
        bbox_center_norm : (2,) normalised image coordinates of bbox centre
            in [-1, 1]. (0, 0) = centred. +x = right, +y = down.
        bbox_size_ratio : bbox diagonal / image diagonal.
        dt : timestep.
        """
        err_pan = bbox_center_norm[0]
        err_tilt = -bbox_center_norm[1]

        desired_pan = self.gimbal.pan + self.cfg.kp_pan * err_pan
        desired_tilt = self.gimbal.tilt + self.cfg.kp_tilt * err_tilt

        self.gimbal.step(desired_pan, desired_tilt, dt)


def project_to_image(
    world_point: NDArray[np.floating],
    camera_pos: NDArray[np.floating],
    gimbal: Gimbal,
    h_fov: float,
    v_fov: float,
    yaw: float = 0.0,
) -> tuple[NDArray[np.floating], bool]:
    """Project a world point into normalised image coordinates.

    Returns
    -------
    (ndc, visible) where ndc is (2,) in [-1, 1] and visible is True
    if the point is in front of the camera and within the FOV.
    """
    R = gimbal.rotation_matrix(yaw)
    local = R.T @ (world_point - camera_pos)

    if local[2] <= 0:
        return np.zeros(2), False

    nx = local[0] / (local[2] * np.tan(h_fov / 2))
    ny = local[1] / (local[2] * np.tan(v_fov / 2))
    ndc = np.array([float(nx), float(ny)])
    visible = bool(abs(nx) <= 1.0 and abs(ny) <= 1.0)
    return ndc, visible
