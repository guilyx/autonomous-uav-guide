# Erwin Lejeune - 2026-02-18
"""Coverage Path Planning (CPP) using boustrophedon decomposition.

Generates lawnmower / boustrophedon scan patterns to achieve full area
coverage at a specified altitude.  The planner accounts for camera/sensor
footprint width and supports rectangular region-of-interest definitions.

Reference: H. Choset, "Coverage of Known Spaces: The Boustrophedon
Cellular Decomposition," Autonomous Robots, 2000.
DOI: 10.1023/A:1008958800904
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CoverageRegion:
    """Axis-aligned rectangular region to cover.

    Parameters
    ----------
    origin : ``[x, y]`` bottom-left corner.
    width : extent in X.
    height : extent in Y.
    altitude : flight altitude for the coverage scan.
    """

    origin: NDArray[np.floating]
    width: float
    height: float
    altitude: float


class CoveragePathPlanner:
    """Boustrophedon (lawnmower) coverage path planner.

    Parameters
    ----------
    swath_width : ground footprint width of the sensor [m].
        For a downward-looking camera this equals
        ``2 * altitude * tan(hfov / 2)`` at the flight altitude.
    overlap : fractional overlap between adjacent sweeps (0..1).
    margin : inset from region boundary [m].
    points_per_row : number of interpolation points per sweep row.
    """

    def __init__(
        self,
        swath_width: float = 5.0,
        overlap: float = 0.1,
        margin: float = 1.0,
        points_per_row: int = 20,
    ) -> None:
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1).")
        self.swath_width = swath_width
        self.overlap = overlap
        self.margin = margin
        self.points_per_row = points_per_row

    def plan(self, region: CoverageRegion) -> NDArray[np.floating]:
        """Compute a boustrophedon coverage path.

        Returns
        -------
        (N, 3) waypoint array at the coverage altitude.
        """
        step = self.swath_width * (1.0 - self.overlap)
        ox, oy = region.origin
        x_lo = ox + self.margin
        x_hi = ox + region.width - self.margin
        y_lo = oy + self.margin
        y_hi = oy + region.height - self.margin

        rows = np.arange(y_lo, y_hi + step * 0.5, step)
        xs = np.linspace(x_lo, x_hi, self.points_per_row)
        alt = region.altitude

        waypoints: list[NDArray[np.floating]] = []
        for i, y in enumerate(rows):
            row_xs = xs if i % 2 == 0 else xs[::-1]
            for x in row_xs:
                waypoints.append(np.array([x, y, alt]))
            # Rounded transition to next row
            if i < len(rows) - 1:
                next_y = rows[i + 1]
                mid_y = 0.5 * (y + next_y)
                end_x = row_xs[-1]
                waypoints.append(np.array([end_x, mid_y, alt]))

        return np.array(waypoints) if waypoints else np.empty((0, 3))

    def estimated_coverage(self, region: CoverageRegion) -> float:
        """Fraction of the region covered by the planned sweeps."""
        step = self.swath_width * (1.0 - self.overlap)
        n_rows = max(1, int(np.ceil((region.height - 2 * self.margin) / step)))
        covered_width = region.width - 2 * self.margin
        covered_area = n_rows * self.swath_width * covered_width
        total_area = region.width * region.height
        return min(1.0, covered_area / total_area)

    @staticmethod
    def swath_from_camera(altitude: float, h_fov: float) -> float:
        """Compute ground swath width from camera parameters.

        Parameters
        ----------
        altitude : flight altitude [m].
        h_fov : horizontal field-of-view [rad].
        """
        return 2.0 * altitude * np.tan(h_fov / 2.0)
