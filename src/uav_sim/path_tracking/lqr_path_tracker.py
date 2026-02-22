# Erwin Lejeune - 2026-02-21
"""LQR-based path tracking controller for quadrotor UAVs.

Extends the hover LQR to track a sequence of waypoints by constructing
a time-varying reference state along the path and applying the
full-state LQR feedback at each step.

Uses sphere-segment intersection (identical to PurePursuit3D) for robust
local look-ahead -- never skipping path segments.

Reference: B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear
Quadratic Methods," Prentice-Hall, 1990, Ch. 8.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D


class LQRPathTracker:
    """LQR controller that follows a 3-D waypoint path.

    At each call the tracker finds a local look-ahead point on the path
    (sphere-segment intersection), builds a full 12-state reference, and
    delegates to the underlying :class:`LQRController`.

    Parameters
    ----------
    lookahead : distance ahead on path to set the reference [m].
    speed : desired cruise speed along the path [m/s].
    mass, gravity, inertia : quadrotor physical parameters.
    """

    def __init__(
        self,
        lookahead: float = 2.0,
        speed: float = 1.5,
        mass: float = 1.5,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
    ) -> None:
        self.lookahead = lookahead
        self.speed = speed
        self._lqr = LQRController(mass=mass, gravity=gravity, inertia=inertia)
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def compute(
        self,
        state: NDArray[np.floating],
        path: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute wrench to track *path* from current *state*.

        Parameters
        ----------
        state : 12-element quadrotor state.
        path : (N, 3) waypoint array.

        Returns
        -------
        ``[T, τx, τy, τz]`` wrench.
        """
        pos = state[:3]
        n = len(path)
        if n == 0:
            return self._lqr.compute(state)

        while self._idx < n - 1:
            if np.linalg.norm(pos - path[self._idx]) < self.lookahead * 0.5:
                self._idx += 1
            else:
                break

        target_pos = self._lookahead_point(pos, path)

        ref = np.zeros(12)
        ref[:3] = target_pos

        direction = target_pos - pos
        dist = np.linalg.norm(direction)
        if dist > 0.01:
            ref[6:9] = direction / dist * self.speed

        return self._lqr.compute(state, target_state=ref)

    def is_path_complete(
        self,
        position: NDArray[np.floating],
        path: NDArray[np.floating],
        threshold: float = 1.0,
    ) -> bool:
        if len(path) == 0:
            return True
        return float(np.linalg.norm(position - path[-1])) < threshold

    def _lookahead_point(
        self, pos: NDArray[np.floating], path: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Sphere-segment intersection search from current segment forward."""
        n = len(path)
        for i in range(self._idx, n - 1):
            pt = PurePursuit3D._intersect_sphere_segment(pos, self.lookahead, path[i], path[i + 1])
            if pt is not None:
                return pt
        return path[min(self._idx, n - 1)].copy()
