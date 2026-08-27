# Erwin Lejeune - 2026-02-17
"""Build occupancy maps from lidar scans using inverse sensor model.

Reference: S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor
Models," Autonomous Robots, 2003.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flybots.costmap.occupancy_grid import OccupancyGrid


class OccupancyMapper:
    """Incremental occupancy mapping from lidar range scans.

    Uses a log-odds representation for efficient Bayesian updates.
    """

    _LOG_ODDS_CLIP = 50.0

    def __init__(
        self,
        grid: OccupancyGrid,
        l_occ: float = 0.85,
        l_free: float = 0.40,
        l0: float = 0.5,
    ) -> None:
        self.grid = grid
        self.l_occ = np.log(l_occ / (1 - l_occ))
        self.l_free = np.log(l_free / (1 - l_free))
        self.l0 = np.log(l0 / (1 - l0))
        self._log_odds = np.full_like(grid.grid, self.l0)

    def update(
        self,
        position: NDArray[np.floating],
        ranges: NDArray[np.floating],
        angles: NDArray[np.floating],
        max_range: float = 30.0,
        yaw: float = 0.0,
        max_range_margin: float = 0.5,
    ) -> None:
        """Integrate one lidar scan into the occupancy grid.

        Parameters
        ----------
        position : world-frame sensor position.
        ranges : measured range per beam [m].
        angles : beam angles **in the body frame** [rad].
        max_range : sensor maximum range [m].
        yaw : vehicle heading [rad].  Beam angles are body-referenced, so
            leaving this at zero integrates every scan as though the
            vehicle were pointing east and smears the map as it turns.
        max_range_margin : ranges within this margin of *max_range* are
            treated as no-return and mark free space only.  Range noise
            pushes a genuine miss just under the limit, which would
            otherwise paint a phantom obstacle at the edge of every scan.
        """
        hit_limit = max_range - max_range_margin
        for r, a in zip(ranges, angles, strict=True):
            direction = np.array([np.cos(yaw + a), np.sin(yaw + a)])
            num_steps = int(min(r, max_range) / self.grid.resolution) + 1
            for step_i in range(num_steps):
                d = step_i * self.grid.resolution
                pt = position[:2] + direction * d
                cell = self.grid.world_to_cell(np.append(pt, 0))[:2]
                if all(0 <= c < s for c, s in zip(cell, self._log_odds.shape[:2], strict=True)):
                    if d < r - self.grid.resolution:
                        self._log_odds[cell] += self.l_free - self.l0
                    elif abs(d - r) < self.grid.resolution and r < hit_limit:
                        self._log_odds[cell] += self.l_occ - self.l0

        # Keep log-odds bounded to avoid overflow in exp() during sigmoid conversion.
        np.clip(self._log_odds, -self._LOG_ODDS_CLIP, self._LOG_ODDS_CLIP, out=self._log_odds)
        prob = 1.0 / (1.0 + np.exp(-self._log_odds))
        if self.grid.dimensions == 2:
            self.grid._grid[:] = prob
        else:
            self.grid._grid[:, :, 0] = prob
