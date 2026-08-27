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


def attitude_from_velocity(
    velocity: NDArray[np.floating],
    acceleration: NDArray[np.floating] | None = None,
    *,
    gravity: float = 9.81,
    max_bank: float = np.pi / 4,
    min_speed: float = 1e-3,
) -> NDArray[np.floating]:
    """Infer a plausible body-to-world rotation from how the vehicle is moving.

    This is a *display* helper, not a state estimate. Many simulations model
    a point mass and carry no attitude at all; drawing those with an identity
    rotation leaves the vehicle stubbornly axis-aligned while it flies a
    curve, which reads as a bug. Deriving heading and bank from the motion
    that *is* simulated keeps the picture honest without inventing dynamics.

    Angles follow the ZYX convention of
    :meth:`~flybots.vehicles.multirotor.multirotor.Multirotor.rotation_matrix`,
    where positive pitch is nose-down and positive roll raises the left wing.

    Parameters
    ----------
    velocity : (3,) world-frame velocity. Sets heading and climb angle.
    acceleration : (3,) world-frame acceleration, optional. Its component
        across the direction of travel sets the coordinated-turn bank; with
        no acceleration the vehicle flies wings-level.
    gravity : magnitude used for the bank balance.
    max_bank : bank limit [rad], so a tight turn cannot roll past knife-edge.
    min_speed : below this the direction of travel is meaningless and the
        identity rotation is returned.

    Returns
    -------
    (3, 3) body-to-world rotation matrix.
    """
    v = np.asarray(velocity, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    if not np.isfinite(speed) or speed < min_speed:
        return np.eye(3)

    yaw = float(np.arctan2(v[1], v[0]))
    pitch = -float(np.arcsin(np.clip(v[2] / speed, -1.0, 1.0)))

    roll = 0.0
    if acceleration is not None:
        a = np.asarray(acceleration, dtype=float).reshape(3)
        horiz = v[:2]
        horiz_speed = float(np.linalg.norm(horiz))
        if horiz_speed > min_speed and np.all(np.isfinite(a)):
            # Left-pointing unit vector in the horizontal plane. The turn
            # acceleration along it is what the bank has to balance.
            left = np.array([-horiz[1], horiz[0]]) / horiz_speed
            a_lat = float(np.dot(a[:2], left))
            # Banking left lowers the left wing, which is negative roll.
            roll = -float(np.arctan2(a_lat, gravity))
            roll = float(np.clip(roll, -max_bank, max_bank))

    cp, sp = np.cos(roll), np.sin(roll)
    ct, st = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
            [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
            [-st, ct * sp, ct * cp],
        ]
    )


def attitude_series_from_positions(
    positions: NDArray[np.floating],
    dt: float,
    **kwargs: Any,
) -> list[NDArray[np.floating]]:
    """Rotation matrices for a whole trajectory of positions.

    Convenience wrapper around :func:`attitude_from_velocity` for the common
    case where a simulation stored positions only. Velocity and acceleration
    come from central differences, so the bank leads and trails a turn the
    way it would if it had been flown.

    Parameters
    ----------
    positions : (N, 3) world-frame positions.
    dt : timestep between samples [s].
    **kwargs : forwarded to :func:`attitude_from_velocity`.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {p.shape}")
    if p.shape[0] < 2 or dt <= 0.0:
        return [np.eye(3) for _ in range(len(p))]
    vel = np.gradient(p, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return [attitude_from_velocity(vel[i], acc[i], **kwargs) for i in range(len(p))]


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
    motor_color: str | None = None,
    motor_size: float = 25.0,
    arm_lw: float = 2.5,
    rotor_disc: bool = True,
    disc_ratio: float = 0.42,
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
    motor_color : Colour of the four motor dots.  Defaults to
        *center_color*; at swarm scale the arms are only a few pixels
        long, so leaving the dots black makes every agent render black
        whatever colour its arms were given.
    motor_size : Marker size for motor dots.
    arm_lw : Line width for arm segments.
    rotor_disc : Draw the swept rotor discs. Without them the model is two
        crossed lines, which at simulation scale reads as a marker rather
        than as an aircraft.
    disc_ratio : Rotor radius as a fraction of *size*.

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
        color=center_color if motor_color is None else motor_color,
        s=motor_size,
        zorder=5,
        depthshade=False,
    )
    arts.append(pt)

    # Rotor discs. Four dots on two crossed lines read as a plus sign at the
    # scale these simulations render at; the swept discs are what make the
    # shape legible as a multirotor rather than a marker.
    if rotor_disc:
        theta = np.linspace(0.0, 2.0 * np.pi, 17)
        r = size * disc_ratio
        circle_body = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
        for tip in (p1, p2, p3, p4):
            pts = circle_body + tip[:3].reshape(3, 1)
            world = R @ pts + np.asarray(position, dtype=float).reshape(3, 1)
            (disc,) = ax.plot(
                world[0],
                world[1],
                world[2],
                color=center_color if motor_color is None else motor_color,
                linewidth=arm_lw * 0.5,
                alpha=0.85,
            )
            arts.append(disc)

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
# Arbitrary rotor layout
# ---------------------------------------------------------------------------


def draw_multirotor_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    rotor_positions: NDArray[np.floating],
    spin_directions: NDArray[np.floating] | None = None,
    scale: float = 1.0,
    arm_color: str = "0.35",
    ccw_color: str = "tab:red",
    cw_color: str = "tab:blue",
    center_color: str = "k",
    motor_size: float = 22.0,
    arm_lw: float = 1.8,
) -> list[Artist]:
    """Draw a multirotor from its actual rotor layout.

    Unlike :func:`draw_quadrotor_3d` and :func:`draw_hexarotor_3d`, which
    assume a symmetric ring, this one takes the rotor positions the model
    is flying with — so an H frame draws as an H and a coaxial pair draws
    as a coaxial pair. Motor dots are coloured by spin direction, which
    makes the pattern that yaw authority comes from visible in the frame.

    Parameters
    ----------
    ax : Axes3D
    position : (3,) world-frame position of the centre of mass.
    R : (3, 3) body-to-world rotation matrix.
    rotor_positions : (n, 3) rotor hubs in body FLU coordinates [m].
    spin_directions : (n,) of +1 (CCW) / -1 (CW), or ``None`` for one colour.
    scale : Multiplier on the body geometry, for legibility at world scale.
    arm_color : Colour of the arm segments.
    ccw_color, cw_color : Motor dot colours per spin direction.
    center_color : Colour of the centre-of-mass marker.
    motor_size : Marker size for motor dots.
    arm_lw : Line width for arm segments.

    Returns
    -------
    List of matplotlib artists.
    """
    hubs = np.asarray(rotor_positions, dtype=np.float64).reshape(-1, 3) * scale
    world = (R @ hubs.T).T + np.asarray(position, dtype=np.float64)

    arts: list[Artist] = []
    for hub in world:
        (arm,) = ax.plot(
            [position[0], hub[0]],
            [position[1], hub[1]],
            [position[2], hub[2]],
            color=arm_color,
            linewidth=arm_lw,
        )
        arts.append(arm)

    if spin_directions is None:
        groups = [(world, center_color)]
    else:
        spins = np.asarray(spin_directions)
        groups = [
            (world[spins > 0], ccw_color),
            (world[spins < 0], cw_color),
        ]

    for points, color in groups:
        if len(points) == 0:
            continue
        arts.append(
            ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                s=motor_size,
                zorder=5,
                depthshade=False,
            )
        )

    arts.append(
        ax.scatter(
            *position,
            color=center_color,
            s=motor_size * 0.7,
            marker="o",
            zorder=6,
            depthshade=False,
        )
    )
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
# VTOL / tilt-rotor
# ---------------------------------------------------------------------------


def draw_vtol_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    tilt: float = 0.0,
    fuselage_length: float = 1.0,
    wingspan: float = 1.6,
    arm_length: float = 0.3,
    scale: float = 1.0,
    body_color: str = "darkslategray",
    wing_color: str = "teal",
    rotor_color: str = "orangered",
    tail_color: str = "slategray",
    lw: float = 2.5,
    **_kw: Any,
) -> list[Artist]:
    """Draw a tilt-rotor VTOL, with the rotors drawn at their actual tilt.

    The tilt is the whole point of the airframe, so it is drawn rather than
    implied: each nacelle's thrust axis rotates in the body x-z plane from
    straight up to straight forward, matching
    :attr:`~flybots.vehicles.vtol.tiltrotor.TiltrotorParams.max_tilt`.

    Parameters
    ----------
    tilt : rotor tilt [rad]. ``0`` is hover (thrust along body ``+z``),
        ``pi/2`` is cruise (thrust along body ``+x``), matching the sign
        convention of :class:`~flybots.vehicles.vtol.tiltrotor.Tiltrotor`.
    """
    fl = fuselage_length * scale
    ws = wingspan * scale
    al = arm_length * scale

    T = _homogeneous_transform(position, R)

    def _line(
        p1: NDArray[np.floating],
        p2: NDArray[np.floating],
        color: str,
        width: float,
    ) -> Artist:
        pw1 = T @ np.array([p1[0], p1[1], p1[2], 1.0])
        pw2 = T @ np.array([p2[0], p2[1], p2[2], 1.0])
        (art,) = ax.plot(
            [pw1[0], pw2[0]],
            [pw1[1], pw2[1]],
            [pw1[2], pw2[2]],
            color=color,
            linewidth=width,
        )
        return art

    arts: list[Artist] = []

    # Fuselage and wing.
    nose = np.array([fl / 2, 0.0, 0.0])
    tail = np.array([-fl / 2, 0.0, 0.0])
    arts.append(_line(nose, tail, body_color, lw))
    wing_l = np.array([fl * 0.05, ws / 2, 0.0])
    wing_r = np.array([fl * 0.05, -ws / 2, 0.0])
    arts.append(_line(wing_l, wing_r, wing_color, lw))

    # V-tail, same silhouette as the fixed-wing artist.
    tail_l = np.array([-fl / 2, ws * 0.15, ws * 0.10])
    tail_r = np.array([-fl / 2, -ws * 0.15, ws * 0.10])
    arts.append(_line(tail, tail_l, tail_color, lw * 0.8))
    arts.append(_line(tail, tail_r, tail_color, lw * 0.8))

    # Four nacelles, ahead of and behind the wing at each semi-span.
    axis = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    for sy in (1.0, -1.0):
        mount = np.array([fl * 0.05, sy * ws * 0.30, 0.0])
        for sx in (1.0, -1.0):
            hub = mount + np.array([sx * al, 0.0, 0.0])
            arts.append(_line(mount, hub, wing_color, lw * 0.6))
            arts.append(_line(hub, hub + axis * al * 0.8, rotor_color, lw * 0.9))
            hub_w = T @ np.array([hub[0], hub[1], hub[2], 1.0])
            arts.append(ax.scatter(*hub_w, color=rotor_color, s=18, zorder=6, depthshade=False))

    nose_w = T @ np.array([nose[0], nose[1], nose[2], 1.0])
    arts.append(ax.scatter(*nose_w, color="red", s=25, zorder=6, depthshade=False))
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
    motor_color: str | None = None,
    center_color: str | None = None,
    rotor_disc: bool = True,
    disc_ratio: float = 0.42,
) -> list[Artist]:
    """Draw a quadrotor top-down cross footprint on a 2D axes.

    Args:
        yaw : Heading [rad]. Pass the direction of travel and the model
            points where the aircraft is going.
        motor_color : Colour of the four motors. Defaults to black, which
            is right for a single vehicle and wrong for a swarm: at fleet
            scale the arms are a few pixels long, so black motors make
            every agent render black whatever colour its arms were given.
        center_color : Colour of the hub. Defaults to *motor_color*.
        rotor_disc : Draw the swept discs. A bare cross reads as a marker
            at these sizes; the discs are what make it read as an aircraft.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])
    motor = "k" if motor_color is None else motor_color
    hub_colour = motor if center_color is None else center_color

    tips = [
        R2 @ np.array([size, 0.0]) + position_xy,
        R2 @ np.array([-size, 0.0]) + position_xy,
        R2 @ np.array([0.0, size]) + position_xy,
        R2 @ np.array([0.0, -size]) + position_xy,
    ]

    arts: list[Artist] = []
    (arm1,) = ax.plot(
        [tips[0][0], tips[1][0]],
        [tips[0][1], tips[1][1]],
        color=arm_colors[0],
        linewidth=arm_lw,
    )
    arts.append(arm1)
    (arm2,) = ax.plot(
        [tips[2][0], tips[3][0]],
        [tips[2][1], tips[3][1]],
        color=arm_colors[1],
        linewidth=arm_lw,
    )
    arts.append(arm2)

    if rotor_disc:
        theta = np.linspace(0.0, 2.0 * np.pi, 17)
        r = size * disc_ratio
        circle = np.stack([r * np.cos(theta), r * np.sin(theta)])
        for tip in tips:
            (disc,) = ax.plot(
                circle[0] + tip[0],
                circle[1] + tip[1],
                color=motor,
                linewidth=arm_lw * 0.55,
                alpha=0.85,
            )
            arts.append(disc)
    else:
        for tip in tips:
            (pt,) = ax.plot(tip[0], tip[1], "o", color=motor, ms=motor_size / 4)
            arts.append(pt)

    (hub,) = ax.plot(*position_xy, "o", color=hub_colour, ms=motor_size / 3)
    arts.append(hub)
    return arts
