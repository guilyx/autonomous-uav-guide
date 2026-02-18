# Erwin Lejeune - 2026-02-15
"""Reusable 3D vehicle drawing primitives for matplotlib.

Each ``draw_*`` function plots a wireframe vehicle model onto an ``Axes3D``
and returns a list of ``matplotlib.artist.Artist`` objects so that the caller
can remove them on the next animation frame (preventing artist accumulation).

The quadrotor is drawn as a cross shape (two perpendicular arms) inspired by
PythonRobotics (Daniel Ingram).  Geometry is specified in body-frame with
homogeneous coordinates, transformed via a 3x4 transformation matrix.

Usage inside an animation ``update`` callback::

    artists: list = []

    def update(frame):
        clear_vehicle_artists(artists)
        pos = positions[frame]
        R   = rotation_matrices[frame]
        artists.extend(draw_quadrotor_3d(ax, pos, R))
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clear_vehicle_artists(artists: list[Artist]) -> None:
    """Remove all artists produced by a previous ``draw_*`` call."""
    while artists:
        a = artists.pop()
        with contextlib.suppress(ValueError, NotImplementedError):
            a.remove()


def _homogeneous_transform(
    position: NDArray[np.floating],
    R: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Build a 3x4 homogeneous transformation matrix ``[R | t]``.

    Parameters
    ----------
    position : (3,) world-frame translation.
    R : (3, 3) body-to-world rotation matrix.

    Returns
    -------
    (3, 4) transformation matrix that maps homogeneous body-frame
    points ``[x, y, z, 1]`` to world-frame ``[x', y', z']``.
    """
    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:3, 3] = position
    return T


# ---------------------------------------------------------------------------
# Quadrotor (cross-arm pattern)
# ---------------------------------------------------------------------------


def draw_quadrotor_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    size: float = 0.25,
    arm_colors: tuple[str, str] = ("red", "blue"),
    center_color: str = "k",
    motor_size: float = 25.0,
    arm_lw: float = 2.5,
    **_kw: Any,
) -> list[Artist]:
    """Draw a quadrotor cross-frame and return the created artists.

    The quadrotor is rendered as two perpendicular arms (a ``+`` shape):
    * Arm 1: ``p1`` <-> ``p2`` along the body-x axis (``red`` by default).
    * Arm 2: ``p3`` <-> ``p4`` along the body-y axis (``blue`` by default).
    Motor positions are shown as dots at each tip.

    Parameters
    ----------
    ax : Axes3D
    position : (3,) world-frame position.
    R : (3, 3) body-to-world rotation matrix.
    size : Half-arm length in world units.
    arm_colors : Colours for arm 1 and arm 2.
    center_color : Colour of the centre-of-mass marker.
    motor_size : Marker size for motor dots.
    arm_lw : Line width for arm segments.

    Returns
    -------
    List of matplotlib artists.
    """
    p1 = np.array([size, 0, 0, 1])
    p2 = np.array([-size, 0, 0, 1])
    p3 = np.array([0, size, 0, 1])
    p4 = np.array([0, -size, 0, 1])

    T = _homogeneous_transform(position, R)
    p1w, p2w, p3w, p4w = T @ p1, T @ p2, T @ p3, T @ p4

    arts: list[Artist] = []

    # Arm 1 (body-x)
    (arm1,) = ax.plot(
        [p1w[0], p2w[0]],
        [p1w[1], p2w[1]],
        [p1w[2], p2w[2]],
        color=arm_colors[0],
        linewidth=arm_lw,
    )
    arts.append(arm1)

    # Arm 2 (body-y)
    (arm2,) = ax.plot(
        [p3w[0], p4w[0]],
        [p3w[1], p4w[1]],
        [p3w[2], p4w[2]],
        color=arm_colors[1],
        linewidth=arm_lw,
    )
    arts.append(arm2)

    # Motor dots at all four tips
    tips = np.column_stack([p1w, p2w, p3w, p4w])
    pt = ax.scatter(
        tips[0],
        tips[1],
        tips[2],
        color="k",
        s=motor_size,
        zorder=5,
        depthshade=False,
    )
    arts.append(pt)

    # Centre hub
    hub = ax.scatter(
        *position,
        color=center_color,
        s=motor_size * 0.6,
        marker="o",
        zorder=6,
        depthshade=False,
    )
    arts.append(hub)

    return arts


# ---------------------------------------------------------------------------
# Hexarotor (three crossing arms)
# ---------------------------------------------------------------------------


def draw_hexarotor_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    size: float = 0.3,
    arm_colors: tuple[str, str] = ("red", "blue"),
    center_color: str = "k",
    motor_size: float = 20.0,
    arm_lw: float = 2.0,
    **_kw: Any,
) -> list[Artist]:
    """Draw a hexarotor frame (3 crossing arms at 60-degree spacing).

    Parameters
    ----------
    See :func:`draw_quadrotor_3d` --- same interface with 6 motors.
    """
    T = _homogeneous_transform(position, R)
    angles = np.linspace(0, np.pi, 3, endpoint=False)
    arts: list[Artist] = []

    for i, a in enumerate(angles):
        p_pos = np.array([size * np.cos(a), size * np.sin(a), 0, 1])
        p_neg = np.array([-size * np.cos(a), -size * np.sin(a), 0, 1])
        pw_pos, pw_neg = T @ p_pos, T @ p_neg
        c = arm_colors[i % 2]
        (line,) = ax.plot(
            [pw_pos[0], pw_neg[0]],
            [pw_pos[1], pw_neg[1]],
            [pw_pos[2], pw_neg[2]],
            color=c,
            linewidth=arm_lw,
        )
        arts.append(line)
        pt = ax.scatter(
            [pw_pos[0], pw_neg[0]],
            [pw_pos[1], pw_neg[1]],
            [pw_pos[2], pw_neg[2]],
            color="k",
            s=motor_size,
            zorder=5,
            depthshade=False,
        )
        arts.append(pt)

    hub = ax.scatter(
        *position,
        color=center_color,
        s=motor_size * 0.8,
        marker="o",
        zorder=6,
        depthshade=False,
    )
    arts.append(hub)
    return arts


# ---------------------------------------------------------------------------
# Fixed-wing
# ---------------------------------------------------------------------------


def draw_fixed_wing_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    fuselage_length: float = 1.0,
    wingspan: float = 1.4,
    scale: float = 1.0,
    body_color: str = "steelblue",
    wing_color: str = "royalblue",
    tail_color: str = "slategray",
    lw: float = 2.5,
    **_kw: Any,
) -> list[Artist]:
    """Draw a simplified fixed-wing wireframe.

    Geometry (body frame, nose at +x):
    * Fuselage: nose -> tail along x.
    * Wing: left tip -> right tip centred at ~30 % from nose.
    * V-tail: two lines from tail upward-left / upward-right.
    """
    fl = fuselage_length * scale
    ws = wingspan * scale

    nose = np.array([fl / 2, 0, 0, 1])
    tail = np.array([-fl / 2, 0, 0, 1])
    wing_l = np.array([fl * 0.05, ws / 2, 0, 1])
    wing_r = np.array([fl * 0.05, -ws / 2, 0, 1])
    tail_l = np.array([-fl / 2, ws * 0.18, ws * 0.12, 1])
    tail_r = np.array([-fl / 2, -ws * 0.18, ws * 0.12, 1])

    T = _homogeneous_transform(position, R)

    def _line(p1_h: NDArray, p2_h: NDArray, color: str) -> Artist:
        pw1, pw2 = T @ p1_h, T @ p2_h
        (art,) = ax.plot(
            [pw1[0], pw2[0]],
            [pw1[1], pw2[1]],
            [pw1[2], pw2[2]],
            color=color,
            linewidth=lw,
        )
        return art

    arts: list[Artist] = []
    arts.append(_line(nose, tail, body_color))
    arts.append(_line(wing_l, wing_r, wing_color))
    arts.append(_line(tail, tail_l, tail_color))
    arts.append(_line(tail, tail_r, tail_color))

    nose_w = T @ nose
    pt = ax.scatter(*nose_w, color="red", s=25, zorder=6, depthshade=False)
    arts.append(pt)
    return arts


# ---------------------------------------------------------------------------
# 2-D footprint helpers (for top-down views)
# ---------------------------------------------------------------------------


def draw_quadrotor_2d(
    ax: Any,
    position_xy: NDArray[np.floating],
    yaw: float,
    size: float = 0.25,
    arm_colors: tuple[str, str] = ("red", "blue"),
    arm_lw: float = 1.5,
    motor_size: float = 15.0,
) -> list[Artist]:
    """Draw a quadrotor top-down cross footprint on a 2D axes."""
    c, s = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])

    p1 = R2 @ np.array([size, 0]) + position_xy
    p2 = R2 @ np.array([-size, 0]) + position_xy
    p3 = R2 @ np.array([0, size]) + position_xy
    p4 = R2 @ np.array([0, -size]) + position_xy

    arts: list[Artist] = []
    (arm1,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=arm_colors[0], linewidth=arm_lw)
    arts.append(arm1)
    (arm2,) = ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=arm_colors[1], linewidth=arm_lw)
    arts.append(arm2)

    for p in [p1, p2, p3, p4]:
        (pt,) = ax.plot(p[0], p[1], "ko", ms=motor_size / 4)
        arts.append(pt)

    (hub,) = ax.plot(*position_xy, "ko", ms=motor_size / 3)
    arts.append(hub)
    return arts
