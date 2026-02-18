# Erwin Lejeune - 2026-02-17
"""2-D / 3-D lidar sensor model via ray-casting against world obstacles.

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT
Press, 2005, Chapter 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import Obstacle
from uav_sim.environment.world import World
from uav_sim.sensors.base import Sensor


class Lidar2D(Sensor):
    """Planar lidar scanner with configurable FOV and angular resolution.

    Returns an array of range measurements.
    """

    def __init__(
        self,
        num_beams: int = 360,
        max_range: float = 30.0,
        fov: float = 2 * np.pi,
        noise_std: float = 0.02,
        rate_hz: float = 10.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(rate_hz, seed)
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
