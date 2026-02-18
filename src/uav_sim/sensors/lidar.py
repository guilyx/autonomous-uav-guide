# Erwin Lejeune - 2026-02-18
"""2-D and 3-D lidar sensor models via ray-casting against world obstacles.

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT
Press, 2005, Chapter 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import Obstacle
from uav_sim.environment.world import World
from uav_sim.sensors.base import Sensor, SensorMount


class Lidar2D(Sensor):
    """Planar lidar scanner with configurable FOV and angular resolution.

    Parameters
    ----------
    num_beams : Number of horizontal rays.
    max_range : Maximum sensing distance [m].
    fov : Horizontal field-of-view [rad] (default 360 deg).
    noise_std : Range measurement noise standard deviation [m].
    """

    def __init__(
        self,
        num_beams: int = 360,
        max_range: float = 30.0,
        fov: float = 2 * np.pi,
        noise_std: float = 0.02,
        rate_hz: float = 10.0,
        seed: int | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        super().__init__(rate_hz, seed, mount)
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov = fov
        self.noise_std = noise_std
        self.angles = np.linspace(-fov / 2, fov / 2, num_beams, endpoint=False)

    def sense(
        self, state: NDArray[np.floating], world: World | None = None
    ) -> NDArray[np.floating]:
        pos = state[:3]
        yaw = state[5] if len(state) > 5 else 0.0
        ranges = np.full(self.num_beams, self.max_range)

        if world is None:
            self._last_measurement = ranges
            return ranges

        for i, angle in enumerate(self.angles):
            direction = np.array([np.cos(yaw + angle), np.sin(yaw + angle), 0.0])
            ranges[i] = self._ray_cast(pos, direction, world.obstacles)

        ranges += self._rng.normal(0, self.noise_std, self.num_beams)
        ranges = np.clip(ranges, 0, self.max_range)
        self._last_measurement = ranges
        return ranges

    def _ray_cast(
        self,
        origin: NDArray[np.floating],
        direction: NDArray[np.floating],
        obstacles: list[Obstacle],
        step: float = 0.1,
    ) -> float:
        """Simple stepped ray-march (can be replaced with analytic intersection)."""
        for d in np.arange(step, self.max_range, step):
            pt = origin + direction * d
            for obs in obstacles:
                if obs.contains(pt):
                    return d
        return self.max_range


class Lidar3D(Sensor):
    """3-D lidar scanner with configurable horizontal and vertical FOV.

    Produces a ``(n_vertical, n_horizontal)`` range image.  The vertical
    scan pattern is typical of multi-line spinning lidars (e.g. Velodyne /
    Ouster) where beams are distributed within ``vertical_fov``.

    Parameters
    ----------
    num_beams_h : Horizontal beams per revolution.
    num_beams_v : Number of vertical channels (laser lines).
    max_range : Maximum sensing distance [m].
    h_fov : Horizontal FOV [rad] (default 360 deg).
    v_fov : Vertical FOV [rad] (default ±15 deg).
    noise_std : Range measurement noise standard deviation [m].
    """

    def __init__(
        self,
        num_beams_h: int = 180,
        num_beams_v: int = 16,
        max_range: float = 30.0,
        h_fov: float = 2 * np.pi,
        v_fov: float = np.radians(30.0),
        noise_std: float = 0.02,
        rate_hz: float = 10.0,
        seed: int | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        super().__init__(rate_hz, seed, mount)
        self.num_beams_h = num_beams_h
        self.num_beams_v = num_beams_v
        self.max_range = max_range
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.noise_std = noise_std
        self.h_angles = np.linspace(-h_fov / 2, h_fov / 2, num_beams_h, endpoint=False)
        self.v_angles = np.linspace(-v_fov / 2, v_fov / 2, num_beams_v, endpoint=True)

    def sense(
        self, state: NDArray[np.floating], world: World | None = None
    ) -> NDArray[np.floating]:
        """Return ``(num_beams_v, num_beams_h)`` range image."""
        pos = state[:3]
        yaw = state[5] if len(state) > 5 else 0.0
        pitch = state[4] if len(state) > 4 else 0.0
        ranges = np.full((self.num_beams_v, self.num_beams_h), self.max_range)

        if world is None:
            self._last_measurement = ranges
            return ranges

        for vi, va in enumerate(self.v_angles):
            cos_va = np.cos(pitch + va)
            sin_va = np.sin(pitch + va)
            for hi, ha in enumerate(self.h_angles):
                dx = cos_va * np.cos(yaw + ha)
                dy = cos_va * np.sin(yaw + ha)
                dz = sin_va
                direction = np.array([dx, dy, dz])
                ranges[vi, hi] = self._ray_cast(pos, direction, world.obstacles)

        ranges += self._rng.normal(0, self.noise_std, ranges.shape)
        ranges = np.clip(ranges, 0, self.max_range)
        self._last_measurement = ranges
        return ranges

    def _ray_cast(
        self,
        origin: NDArray[np.floating],
        direction: NDArray[np.floating],
        obstacles: list[Obstacle],
        step: float = 0.2,
    ) -> float:
        for d in np.arange(step, self.max_range, step):
            pt = origin + direction * d
            for obs in obstacles:
                if obs.contains(pt):
                    return d
        return self.max_range

    def to_point_cloud(
        self,
        state: NDArray[np.floating],
        ranges: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Convert the range image to an ``(N, 3)`` world-frame point cloud.

        Only returns points with ``range < max_range``.
        """
        if ranges is None:
            ranges = self._last_measurement
        if ranges is None:
            return np.zeros((0, 3))

        pos = state[:3]
        yaw = state[5] if len(state) > 5 else 0.0
        pitch = state[4] if len(state) > 4 else 0.0

        pts: list[NDArray[np.floating]] = []
        for vi, va in enumerate(self.v_angles):
            cos_va = np.cos(pitch + va)
            sin_va = np.sin(pitch + va)
            for hi, ha in enumerate(self.h_angles):
                r = ranges[vi, hi]
                if r >= self.max_range - 1e-3:
                    continue
                dx = r * cos_va * np.cos(yaw + ha)
                dy = r * cos_va * np.sin(yaw + ha)
                dz = r * sin_va
                pts.append(pos + np.array([dx, dy, dz]))

        return np.array(pts) if pts else np.zeros((0, 3))
