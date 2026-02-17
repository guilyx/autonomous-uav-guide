# Erwin Lejeune - 2026-02-15
"""Reusable 3D vehicle drawing primitives for matplotlib.

Each ``draw_*`` function plots a wireframe vehicle model onto an ``Axes3D``
and returns a list of ``matplotlib.artist.Artist`` objects so that the caller
can remove them on the next animation frame (preventing artist accumulation).

Usage inside an animation ``update`` callback::

    artists: list = []

    def update(frame):
        clear_vehicle_artists(artists)
        pos = positions[frame]
        R   = rotation_matrices[frame]
        artists.extend(draw_quadrotor_3d(ax, pos, R))
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def clear_vehicle_artists(artists: list[Artist]) -> None:
    """Remove all artists produced by a previous ``draw_*`` call."""
    import contextlib

    while artists:
        a = artists.pop()
        with contextlib.suppress(ValueError, NotImplementedError):
            a.remove()


def _transform(
    body_pts: NDArray[np.floating],
    position: NDArray[np.floating],
    R: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Apply rotation *R* and translation *position* to body-frame points.

    Parameters
    ----------
    body_pts : (3, N)
        Columns are body-frame 3-D coordinates.
    position : (3,)
        World-frame position.
    R : (3, 3)
        Body-to-world rotation matrix.

    Returns
    -------
    (3, N) world-frame coordinates.
    """
    return R @ body_pts + position.reshape(3, 1)


# ---------------------------------------------------------------------------
# Quadrotor
# ---------------------------------------------------------------------------


def draw_quadrotor_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    arm_length: float = 0.0397,
    scale: float = 10.0,
    colors: tuple[str, str] | None = None,
    center_color: str = "k",
    motor_size: float = 20.0,
    arm_lw: float = 2.0,
    **_kw: Any,
) -> list[Artist]:
    """Draw a quadrotor X-frame and return the created artists.

    Parameters
    ----------
    ax : Axes3D
    position : (3,) world-frame position.
    R : (3, 3) body-to-world rotation matrix.
    arm_length : Physical arm length (m).
    scale : Visual scaling factor.
    colors : Two alternating arm colours, default ``("red", "blue")``.
    center_color : Colour of the centre-of-mass marker.
    motor_size : Marker size for motor dots.
    arm_lw : Line width for arm segments.

    Returns
    -------
    List of matplotlib artists.
    """
    if colors is None:
        colors = ("red", "blue")

    L = arm_length * scale
    s = L / np.sqrt(2.0)

    # Body-frame motor positions (X-configuration: 45°, 135°, 225°, 315°)
    motors_body = np.array(
        [
            [s, s, 0],
            [s, -s, 0],
            [-s, -s, 0],
            [-s, s, 0],
        ]
    ).T  # (3, 4)

    motors_w = _transform(motors_body, position, R)
    arts: list[Artist] = []

    arm_colors = [colors[0], colors[1], colors[0], colors[1]]
    for i in range(4):
        (line,) = ax.plot(
            [position[0], motors_w[0, i]],
            [position[1], motors_w[1, i]],
            [position[2], motors_w[2, i]],
            color=arm_colors[i],
            linewidth=arm_lw,
        )
        arts.append(line)
        pt = ax.scatter(
            motors_w[0, i],
            motors_w[1, i],
            motors_w[2, i],
            color=arm_colors[i],
            s=motor_size,
            zorder=5,
        )
        arts.append(pt)

    # Centre hub
    hub = ax.scatter(
        *position,
        color=center_color,
        s=motor_size * 1.5,
        marker="o",
        zorder=6,
    )
    arts.append(hub)
    return arts


# ---------------------------------------------------------------------------
# Hexarotor
# ---------------------------------------------------------------------------


def draw_hexarotor_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    arm_length: float = 0.06,
    scale: float = 10.0,
    colors: tuple[str, str] | None = None,
    center_color: str = "k",
    motor_size: float = 18.0,
    arm_lw: float = 2.0,
    **_kw: Any,
) -> list[Artist]:
    """Draw a hexarotor frame (6 arms at 60-degree spacing).

    Parameters
    ----------
    See :func:`draw_quadrotor_3d` — same interface with 6 motors.
    """
    if colors is None:
        colors = ("red", "blue")

    L = arm_length * scale
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 0, 60, 120, …, 300 deg

    motors_body = np.array([[L * np.cos(a), L * np.sin(a), 0.0] for a in angles]).T  # (3, 6)

    motors_w = _transform(motors_body, position, R)
    arts: list[Artist] = []

    for i in range(6):
        c = colors[i % 2]
        (line,) = ax.plot(
            [position[0], motors_w[0, i]],
            [position[1], motors_w[1, i]],
            [position[2], motors_w[2, i]],
            color=c,
            linewidth=arm_lw,
        )
        arts.append(line)
        pt = ax.scatter(
            motors_w[0, i],
            motors_w[1, i],
            motors_w[2, i],
            color=c,
            s=motor_size,
            zorder=5,
        )
        arts.append(pt)

    hub = ax.scatter(
        *position,
        color=center_color,
        s=motor_size * 1.5,
        marker="o",
        zorder=6,
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
    * Fuselage: nose → tail along x.
    * Wing: left tip → right tip centred at ~30 % from nose.
    * V-tail: two lines from tail upward-left / upward-right.

    Parameters
    ----------
    ax : Axes3D
    position, R : pose.
    fuselage_length, wingspan, scale : sizing.
    body_color, wing_color, tail_color : colours.
    lw : line width.
    """
    fl = fuselage_length * scale
    ws = wingspan * scale

    nose = np.array([fl / 2, 0, 0])
    tail = np.array([-fl / 2, 0, 0])
    wing_l = np.array([fl * 0.05, ws / 2, 0])
    wing_r = np.array([fl * 0.05, -ws / 2, 0])
    tail_l = np.array([-fl / 2, ws * 0.18, ws * 0.12])
    tail_r = np.array([-fl / 2, -ws * 0.18, ws * 0.12])

    def _line(p1: NDArray, p2: NDArray, color: str) -> Artist:
        pts = np.column_stack([p1, p2])  # (3, 2) body
        pw = _transform(pts, position, R)
        (art,) = ax.plot(pw[0], pw[1], pw[2], color=color, linewidth=lw)
        return art

    arts: list[Artist] = []
    # Fuselage
    arts.append(_line(nose, tail, body_color))
    # Wings
    arts.append(_line(wing_l, wing_r, wing_color))
    # V-tail
    arts.append(_line(tail, tail_l, tail_color))
    arts.append(_line(tail, tail_r, tail_color))
    # Nose dot
    nose_w = _transform(nose.reshape(3, 1), position, R).ravel()
    pt = ax.scatter(*nose_w, color="red", s=25, zorder=6)
    arts.append(pt)
    return arts


# ---------------------------------------------------------------------------
# 2-D footprint helpers (for top-down views)
# ---------------------------------------------------------------------------


def draw_quadrotor_2d(
    ax: Any,
    position_xy: NDArray[np.floating],
    yaw: float,
    arm_length: float = 0.0397,
    scale: float = 10.0,
    colors: tuple[str, str] | None = None,
    arm_lw: float = 1.5,
    motor_size: float = 15.0,
) -> list[Artist]:
    """Draw a quadrotor top-down footprint on a 2D axes."""
    if colors is None:
        colors = ("red", "blue")

    L = arm_length * scale
    s = L / np.sqrt(2.0)
    c, sn = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, -sn], [sn, c]])

    motors_body = np.array([[s, s], [s, -s], [-s, -s], [-s, s]])
    motors_w = (R2 @ motors_body.T).T + position_xy

    arts: list[Artist] = []
    for i in range(4):
        col = colors[i % 2]
        (line,) = ax.plot(
            [position_xy[0], motors_w[i, 0]],
            [position_xy[1], motors_w[i, 1]],
            color=col,
            linewidth=arm_lw,
        )
        arts.append(line)
        (pt,) = ax.plot(motors_w[i, 0], motors_w[i, 1], "o", color=col, ms=motor_size / 4)
        arts.append(pt)

    (hub,) = ax.plot(*position_xy, "ko", ms=motor_size / 3)
    arts.append(hub)
    return arts
