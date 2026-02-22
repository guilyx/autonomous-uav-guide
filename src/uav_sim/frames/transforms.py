# Erwin Lejeune - 2026-02-15
"""3-D coordinate frame transformations for UAV systems.

Defines conversions between three reference frames:

* **World (ENU)** — fixed East-North-Up global frame.
* **Body (FCU)** — Forward-Left-Up frame attached to the vehicle CoG,
  rotated by the vehicle's ZYX Euler angles (yaw, pitch, roll).
* **Sensor** — arbitrary offset from the body frame described by a
  :class:`~uav_sim.sensors.base.SensorMount`.

All transforms are pure rotation + translation using 3×3 matrices,
matching the convention of :meth:`Quadrotor.rotation_matrix` (body → world).

Inspired by the ``simple_autonomous_car`` frames module, extended to 3-D.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class FrameID(Enum):
    """Identifier for a coordinate frame."""

    WORLD = auto()
    BODY = auto()
    SENSOR = auto()


# ---------------------------------------------------------------------------
# Rotation helpers (ZYX Euler — same convention as Quadrotor)
# ---------------------------------------------------------------------------


def euler_to_rotation(phi: float, theta: float, psi: float) -> NDArray[np.floating]:
    """ZYX Euler angles → 3×3 rotation matrix (body → world).

    Parameters
    ----------
    phi : Roll angle [rad].
    theta : Pitch angle [rad].
    psi : Yaw angle [rad].
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
            [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
            [-st, ct * sp, ct * cp],
        ]
    )


# ---------------------------------------------------------------------------
# Single-point transforms
# ---------------------------------------------------------------------------


def body_to_world(
    point_body: NDArray[np.floating],
    position: NDArray[np.floating],
    euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform a point from body frame to world frame.

    Parameters
    ----------
    point_body : (3,) point in body frame.
    position : (3,) vehicle position in world frame.
    euler : (3,) [roll, pitch, yaw] of the vehicle.
    """
    R = euler_to_rotation(*euler)
    return (R @ np.asarray(point_body)) + np.asarray(position)


def world_to_body(
    point_world: NDArray[np.floating],
    position: NDArray[np.floating],
    euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform a point from world frame to body frame.

    Parameters
    ----------
    point_world : (3,) point in world frame.
    position : (3,) vehicle position in world frame.
    euler : (3,) [roll, pitch, yaw] of the vehicle.
    """
    R = euler_to_rotation(*euler)
    return R.T @ (np.asarray(point_world) - np.asarray(position))


def sensor_to_body(
    point_sensor: NDArray[np.floating],
    mount_position: NDArray[np.floating],
    mount_orientation: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform a point from sensor frame to body frame.

    Parameters
    ----------
    point_sensor : (3,) point in sensor frame.
    mount_position : (3,) sensor offset in body frame [x, y, z].
    mount_orientation : (3,) sensor Euler angles in body frame [r, p, y].
    """
    R_sb = euler_to_rotation(*mount_orientation)
    return (R_sb @ np.asarray(point_sensor)) + np.asarray(mount_position)


def body_to_sensor(
    point_body: NDArray[np.floating],
    mount_position: NDArray[np.floating],
    mount_orientation: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform a point from body frame to sensor frame.

    Parameters
    ----------
    point_body : (3,) point in body frame.
    mount_position : (3,) sensor offset in body frame [x, y, z].
    mount_orientation : (3,) sensor Euler angles in body frame [r, p, y].
    """
    R_sb = euler_to_rotation(*mount_orientation)
    return R_sb.T @ (np.asarray(point_body) - np.asarray(mount_position))


# ---------------------------------------------------------------------------
# Convenience chains
# ---------------------------------------------------------------------------


def sensor_to_world(
    point_sensor: NDArray[np.floating],
    mount_position: NDArray[np.floating],
    mount_orientation: NDArray[np.floating],
    vehicle_position: NDArray[np.floating],
    vehicle_euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Sensor frame → body frame → world frame (convenience chain)."""
    body = sensor_to_body(point_sensor, mount_position, mount_orientation)
    return body_to_world(body, vehicle_position, vehicle_euler)


def world_to_sensor(
    point_world: NDArray[np.floating],
    mount_position: NDArray[np.floating],
    mount_orientation: NDArray[np.floating],
    vehicle_position: NDArray[np.floating],
    vehicle_euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """World frame → body frame → sensor frame (convenience chain)."""
    body = world_to_body(point_world, vehicle_position, vehicle_euler)
    return body_to_sensor(body, mount_position, mount_orientation)


# ---------------------------------------------------------------------------
# Batch transforms (N, 3) arrays
# ---------------------------------------------------------------------------


def batch_body_to_world(
    points_body: NDArray[np.floating],
    position: NDArray[np.floating],
    euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform (N, 3) points from body frame to world frame."""
    R = euler_to_rotation(*euler)
    return (points_body @ R.T) + np.asarray(position)


def batch_world_to_body(
    points_world: NDArray[np.floating],
    position: NDArray[np.floating],
    euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform (N, 3) points from world frame to body frame."""
    R = euler_to_rotation(*euler)
    return (points_world - np.asarray(position)) @ R


def batch_sensor_to_world(
    points_sensor: NDArray[np.floating],
    mount_position: NDArray[np.floating],
    mount_orientation: NDArray[np.floating],
    vehicle_position: NDArray[np.floating],
    vehicle_euler: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Transform (N, 3) points from sensor frame to world frame."""
    R_sb = euler_to_rotation(*mount_orientation)
    points_body = (points_sensor @ R_sb.T) + np.asarray(mount_position)
    R_bw = euler_to_rotation(*vehicle_euler)
    return (points_body @ R_bw.T) + np.asarray(vehicle_position)
