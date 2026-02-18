# Erwin Lejeune - 2026-02-18
"""Pinhole camera model with projective geometry and FOV computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.base import Sensor, SensorMount


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    @property
    def K(self) -> NDArray[np.floating]:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @property
    def h_fov(self) -> float:
        """Horizontal field-of-view [rad]."""
        return 2.0 * np.arctan(self.width / (2.0 * self.fx))

    @property
    def v_fov(self) -> float:
        """Vertical field-of-view [rad]."""
        return 2.0 * np.arctan(self.height / (2.0 * self.fy))

    @property
    def diagonal_fov(self) -> float:
        """Diagonal field-of-view [rad]."""
        half_diag = np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2)
        f_avg = (self.fx + self.fy) / 2.0
        return 2.0 * np.arctan(half_diag / f_avg)


class Camera(Sensor):
    """Pinhole camera with projective geometry, FOV, and frustum helpers.

    Parameters
    ----------
    intrinsics : Camera intrinsic parameters.
    max_depth : Maximum sensing depth [m] for frustum visualisation.
    mount : How the camera is mounted on the vehicle body.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        max_depth: float = 15.0,
        rate_hz: float = 30.0,
        seed: int | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        super().__init__(rate_hz, seed, mount)
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.max_depth = max_depth

    @property
    def h_fov(self) -> float:
        return self.intrinsics.h_fov

    @property
    def v_fov(self) -> float:
        return self.intrinsics.v_fov

    def frustum_corners(
        self,
        position: NDArray[np.floating],
        R: NDArray[np.floating],
        depth: float | None = None,
    ) -> NDArray[np.floating]:
        """Return the 4 far-plane corners of the camera frustum in world frame.

        Returns shape ``(4, 3)`` in order: top-left, top-right,
        bottom-right, bottom-left (looking along +Z camera axis).
        """
        d = depth if depth is not None else self.max_depth
        half_h = d * np.tan(self.intrinsics.h_fov / 2)
        half_v = d * np.tan(self.intrinsics.v_fov / 2)

        corners_cam = np.array(
            [
                [-half_h, -half_v, d],
                [half_h, -half_v, d],
                [half_h, half_v, d],
                [-half_h, half_v, d],
            ]
        )
        R_mount = self.mount.rotation_matrix()
        return (R @ R_mount @ corners_cam.T).T + position

    def project(
        self,
        points_world: NDArray[np.floating],
        camera_pose: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Project (N, 3) world points to (N, 2) pixel coordinates."""
        K = self.intrinsics.K
        pos = camera_pose[:3]
        pts_cam = points_world - pos
        uv_h = (K @ pts_cam.T).T
        depth = uv_h[:, 2:]
        depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        return uv_h[:, :2] / depth

    def sense(self, state: NDArray[np.floating], world=None) -> NDArray[np.floating]:
        self._last_measurement = state[:3].copy()
        return self._last_measurement
