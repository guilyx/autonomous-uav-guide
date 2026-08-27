# Erwin Lejeune - 2026-02-16
"""Virtual structure formation control.

Reference: M. A. Lewis, K.-H. Tan, "High Precision Formation Control of
Mobile Robots Using Virtual Structures," Autonomous Robots, 1997.
DOI: 10.1023/A:1008814708459
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from flybots.vehicles.multirotor.quadrotor import Quadrotor


class VirtualStructure:
    """Virtual rigid body formation controller.

    Defines a virtual body with position and orientation. Each agent
    tracks its assigned fixed offset in the virtual body frame.

    Parameters:
        body_offsets: (N, 3) offsets of each agent in the virtual body frame.
        kp: Position tracking gain.
        kd: Velocity damping gain.
    """

    def __init__(
        self,
        body_offsets: NDArray[np.floating],
        kp: float = 3.0,
        kd: float = 2.0,
    ) -> None:
        self.body_offsets = np.asarray(body_offsets, dtype=np.float64)
        self.N = len(body_offsets)
        self.kp = kp
        self.kd = kd

    def desired_positions(
        self,
        body_pos: NDArray[np.floating],
        body_yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """Compute desired world positions for all agents.

        Args:
            body_pos: Virtual body position ``[x, y, z]``.
            body_yaw: Virtual body yaw [rad].

        Returns:
            (N, 3) desired positions.
        """
        R = Quadrotor.rotation_matrix(0.0, 0.0, body_yaw)
        return body_pos + (R @ self.body_offsets.T).T

    def compute_forces(
        self,
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        body_pos: NDArray[np.floating],
        body_yaw: float = 0.0,
        body_vel: NDArray[np.floating] | None = None,
        body_yaw_rate: float = 0.0,
        body_acc: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Compute PD tracking forces towards desired formation positions.

        Args:
            positions: (N, 3) current agent positions.
            velocities: (N, 3) current agent velocities.
            body_pos: Virtual body position.
            body_yaw: Virtual body yaw.
            body_vel: Virtual body velocity.  Damping against the world
                frame instead of against the *slot* velocity leaves a
                standing formation error proportional to how fast the
                structure is moving; feeding the slot velocity in removes
                it.
            body_yaw_rate: Virtual body yaw rate [rad/s], used to add the
                tangential ``ω × r`` component of each slot's velocity.
            body_acc: Virtual body acceleration, added straight through as
                feed-forward so the PD only has to correct the residual.

        Returns:
            (N, 3) control forces.
        """
        des = self.desired_positions(body_pos, body_yaw)
        if body_vel is None:
            return self.kp * (des - positions) - self.kd * velocities

        arm = des - body_pos
        des_vel = np.broadcast_to(np.asarray(body_vel, dtype=float), (self.N, 3)).copy()
        omega = np.array([0.0, 0.0, body_yaw_rate])
        if body_yaw_rate != 0.0:
            des_vel += np.cross(omega, arm)

        forces = self.kp * (des - positions) + self.kd * (des_vel - velocities)
        if body_acc is not None:
            des_acc = np.broadcast_to(np.asarray(body_acc, dtype=float), (self.N, 3)).copy()
            if body_yaw_rate != 0.0:
                # Centripetal term of a slot rotating with the structure.
                des_acc += np.cross(omega, np.cross(omega, arm))
            forces = forces + des_acc
        return forces
