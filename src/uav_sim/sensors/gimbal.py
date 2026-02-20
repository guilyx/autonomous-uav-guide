# Erwin Lejeune - 2026-02-18
"""Two-axis gimbal model with first-order rate dynamics.

Models a pan-tilt gimbal that can point a camera at arbitrary targets.
The gimbal state is (pan, tilt) in radians, and the controller accepts
target angles or a look-at point in world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GimbalLimits:
    """Joint limits for pan/tilt axes [rad]."""

    pan_min: float = -np.pi
    pan_max: float = np.pi
    tilt_min: float = -np.pi / 2
    tilt_max: float = np.pi / 4


class Gimbal:
    """Two-axis pan-tilt gimbal with rate-limited dynamics.

    Parameters
    ----------
    max_rate : maximum angular rate per axis [rad/s].
    limits : joint-angle limits.
    """

    def __init__(
        self,
        max_rate: float = 1.5,
        limits: GimbalLimits | None = None,
    ) -> None:
        self.max_rate = max_rate
        self.limits = limits or GimbalLimits()
        self.pan: float = 0.0
        self.tilt: float = -np.pi / 4  # default: looking 45° down

    def reset(self, pan: float = 0.0, tilt: float = -np.pi / 4) -> None:
        self.pan = pan
        self.tilt = tilt

    @property
    def angles(self) -> NDArray[np.floating]:
        """Current ``[pan, tilt]`` in radians."""
        return np.array([self.pan, self.tilt])

    def look_at(
        self,
        camera_pos: NDArray[np.floating],
        target: NDArray[np.floating],
        yaw: float = 0.0,
    ) -> tuple[float, float]:
        """Compute desired pan/tilt to point at *target* from *camera_pos*.

        Parameters
        ----------
        camera_pos : world-frame position of the gimbal pivot.
        target : world-frame point to look at.
        yaw : vehicle heading [rad] (gimbal pan is relative to heading).

        Returns
        -------
        (desired_pan, desired_tilt) in radians.
        """
        delta = target - camera_pos
        dx, dy, dz = delta[0], delta[1], delta[2]
        desired_pan = np.arctan2(dy, dx) - yaw
        horiz = np.sqrt(dx**2 + dy**2)
        desired_tilt = np.arctan2(dz, horiz)
        return float(desired_pan), float(desired_tilt)

    def step(self, desired_pan: float, desired_tilt: float, dt: float) -> None:
        """Advance gimbal state towards desired angles with rate limiting."""
        dp = np.clip(desired_pan - self.pan, -self.max_rate * dt, self.max_rate * dt)
        dt_ = np.clip(desired_tilt - self.tilt, -self.max_rate * dt, self.max_rate * dt)
        self.pan = float(np.clip(self.pan + dp, self.limits.pan_min, self.limits.pan_max))
        self.tilt = float(np.clip(self.tilt + dt_, self.limits.tilt_min, self.limits.tilt_max))

    def rotation_matrix(self, yaw: float = 0.0) -> NDArray[np.floating]:
        """3x3 rotation from gimbal (camera) frame to world frame.

        Camera frame: +Z = optical axis, +X = right, +Y = down.
        World frame: +Z = up.

        The optical axis direction in world is
        ``[cos(h)cos(t), sin(h)cos(t), sin(t)]`` where ``h = yaw + pan``
        and ``t = tilt`` (negative tilt → looking down).
        """
        heading = yaw + self.pan
        ch, sh = np.cos(heading), np.sin(heading)
        ct, st = np.cos(self.tilt), np.sin(self.tilt)
        return np.array(
            [
                [-sh, ch * st, ch * ct],
                [ch, sh * st, sh * ct],
                [0.0, -ct, st],
            ]
        )

    def frustum_corners_world(
        self,
        camera_pos: NDArray[np.floating],
        h_fov: float,
        v_fov: float,
        depth: float,
        yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """4 far-plane corners of the camera frustum in world frame.

        Returns (4, 3) array: TL, TR, BR, BL looking along optical axis.
        """
        half_h = depth * np.tan(h_fov / 2)
        half_v = depth * np.tan(v_fov / 2)
        corners_cam = np.array(
            [
                [-half_h, -half_v, depth],
                [half_h, -half_v, depth],
                [half_h, half_v, depth],
                [-half_h, half_v, depth],
            ]
        )
        R = self.rotation_matrix(yaw)
        return (R @ corners_cam.T).T + camera_pos
