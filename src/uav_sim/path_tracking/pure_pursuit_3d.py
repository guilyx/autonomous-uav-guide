# Erwin Lejeune - 2026-02-17
"""3-D Pure-Pursuit path tracker for multirotor UAVs.

Adapts the classic look-ahead pure-pursuit algorithm to 3-D waypoint
sequences.  At each step, the tracker finds the look-ahead point on the
path and returns it as the target position for a lower-level controller.

The ``adaptive`` option automatically scales the look-ahead distance
with speed to remain conservative at high velocity.

Reference: R. C. Coulter, "Implementation of the Pure Pursuit Path
Tracking Algorithm," CMU-RI-TR-92-01, 1992.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PurePursuit3D:
    """Pure-pursuit look-ahead tracker in three dimensions.

    Parameters
    ----------
    lookahead : float
        Base look-ahead distance [m].
    waypoint_threshold : float
        Distance at which the tracker advances to the next segment.
    speed : float
        Desired forward speed along the path [m/s].
    adaptive : bool
        If True, scale look-ahead with speed for smoother conservative tracking.
    """

    def __init__(
        self,
        lookahead: float = 5.0,
        waypoint_threshold: float = 0.2,
        speed: float = 1.0,
        *,
        adaptive: bool = True,
    ) -> None:
        self.lookahead = lookahead
        self.waypoint_threshold = waypoint_threshold
        self.speed = speed
        self.adaptive = adaptive
        self._idx = 0

    def reset(self) -> None:
        """Reset internal waypoint index."""
        self._idx = 0

    @property
    def current_index(self) -> int:
        """Index of the *target* waypoint being pursued."""
        return self._idx

    def compute_target(
        self,
        position: NDArray[np.floating],
        path: NDArray[np.floating],
        velocity: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Return the look-ahead target point on *path*.

        Parameters
        ----------
        position : (3,) current UAV position.
        path : (N, 3) ordered waypoint sequence.
        velocity : (3,) optional current velocity for adaptive look-ahead.

        Returns
        -------
        (3,) target position.
        """
        n = len(path)
        if n == 0:
            return position.copy()

        # Advance segment index when close to current waypoint
        while self._idx < n - 1:
            if float(np.linalg.norm(position - path[self._idx])) < self.waypoint_threshold:
                self._idx += 1
            else:
                break

        # Adaptive look-ahead: scale with speed for conservative tracking
        la = self.lookahead
        if self.adaptive and velocity is not None:
            spd = float(np.linalg.norm(velocity))
            la = max(self.lookahead, self.lookahead * (1.0 + 0.3 * spd))

        # Search for look-ahead point along remaining path
        for i in range(self._idx, n - 1):
            pt = self._intersect_sphere_segment(position, la, path[i], path[i + 1])
            if pt is not None:
                return pt

        # Fall back: target the current waypoint
        return path[min(self._idx, n - 1)].copy()

    def is_path_complete(
        self,
        position: NDArray[np.floating],
        path: NDArray[np.floating],
    ) -> bool:
        """Check if the UAV has reached the final waypoint."""
        if len(path) == 0:
            return True
        return float(np.linalg.norm(position - path[-1])) < self.waypoint_threshold

    @staticmethod
    def _intersect_sphere_segment(
        center: NDArray[np.floating],
        radius: float,
        p1: NDArray[np.floating],
        p2: NDArray[np.floating],
    ) -> NDArray[np.floating] | None:
        """Furthest intersection of a sphere with a line segment, or None."""
        d = p2 - p1
        f = p1 - center
        a = float(d @ d)
        b = float(2.0 * f @ d)
        c_val = float(f @ f) - radius * radius

        disc = b * b - 4 * a * c_val
        if disc < 0 or a < 1e-12:
            return None

        sqrt_disc = np.sqrt(disc)
        t2 = (-b + sqrt_disc) / (2 * a)
        t1 = (-b - sqrt_disc) / (2 * a)

        for t in [t2, t1]:
            if 0.0 <= t <= 1.0:
                return p1 + t * d
        return None
