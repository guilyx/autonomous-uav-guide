# Erwin Lejeune - 2026-02-17
"""Pinhole camera model with projective geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.base import Sensor


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


class Camera(Sensor):
    """Downward or forward-looking camera that projects 3-D points to pixels."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        rate_hz: float = 30.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(rate_hz, seed)
        self.intrinsics = intrinsics or CameraIntrinsics()

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
