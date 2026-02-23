# Erwin Lejeune - 2026-02-18
"""Sensor field-of-view (FOV) visualisation helpers.

Draws lidar FOV wedges / cones and camera frustums on 3D, top-down (XY),
and side (XZ) matplotlib axes.  All functions return a list of artists so
callers can remove them on the next animation frame.

Typical usage inside an animation loop::

    fov_arts: list = []

    def update(frame):
        clear_vehicle_artists(fov_arts)
        fov_arts.extend(draw_lidar2d_fov(ax3d, pos, yaw, lidar))
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.patches import Polygon, Wedge
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from numpy.typing import NDArray

from uav_sim.visualization.theme import C_CAMERA, C_LIDAR, C_LIDAR3D, C_LIDAR_HIT

# ── 2-D lidar FOV ────────────────────────────────────────────────────────────


def draw_lidar2d_fov_top(
    ax: Any,
    position_xy: NDArray[np.floating],
    yaw: float,
    fov: float = 2 * np.pi,
    max_range: float = 12.0,
    *,
    color: str = C_LIDAR,
    alpha: float = 0.08,
    edge_alpha: float = 0.3,
) -> list[Artist]:
    """Draw the 2D lidar sensing wedge on a top-down (XY) axes.

    For a full-circle lidar the wedge becomes a filled circle.
    """
    arts: list[Artist] = []
    if fov >= 2 * np.pi - 1e-3:
        circle = plt_Circle(position_xy, max_range, color=color, alpha=alpha, lw=0)
        ax.add_patch(circle)
        arts.append(circle)
        edge = plt_Circle(
            position_xy, max_range, fill=False, edgecolor=color, alpha=edge_alpha, lw=0.6
        )
        ax.add_patch(edge)
        arts.append(edge)
    else:
        start_deg = np.degrees(yaw - fov / 2)
        wedge = Wedge(
            position_xy,
            max_range,
            start_deg,
            start_deg + np.degrees(fov),
            color=color,
            alpha=alpha,
            lw=0,
        )
        ax.add_patch(wedge)
        arts.append(wedge)
        wedge_edge = Wedge(
            position_xy,
            max_range,
            start_deg,
            start_deg + np.degrees(fov),
            fill=False,
            edgecolor=color,
            alpha=edge_alpha,
            lw=0.6,
        )
        ax.add_patch(wedge_edge)
        arts.append(wedge_edge)
    return arts


def draw_lidar2d_rays_top(
    ax: Any,
    position_xy: NDArray[np.floating],
    yaw: float,
    ranges: NDArray[np.floating],
    angles: NDArray[np.floating],
    max_range: float = 12.0,
    *,
    ray_color: str = C_LIDAR,
    hit_color: str = C_LIDAR_HIT,
    ray_alpha: float = 0.25,
    ray_lw: float = 0.4,
    hit_size: float = 8.0,
    every_n: int = 1,
) -> list[Artist]:
    """Draw individual lidar rays and hit points on a top-down axes."""
    arts: list[Artist] = []
    hit_x, hit_y = [], []
    for i in range(0, len(ranges), every_n):
        a = yaw + angles[i]
        r = ranges[i]
        end = position_xy + r * np.array([np.cos(a), np.sin(a)])
        (ray,) = ax.plot(
            [position_xy[0], end[0]],
            [position_xy[1], end[1]],
            color=ray_color,
            lw=ray_lw,
            alpha=ray_alpha,
        )
        arts.append(ray)
        if r < max_range - 0.5:
            hit_x.append(end[0])
            hit_y.append(end[1])
    if hit_x:
        sc = ax.scatter(hit_x, hit_y, c=hit_color, s=hit_size, zorder=8, alpha=0.7)
        arts.append(sc)
    return arts


def draw_lidar2d_fov_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    yaw: float,
    fov: float = 2 * np.pi,
    max_range: float = 12.0,
    *,
    color: str = C_LIDAR,
    alpha: float = 0.06,
    n_segments: int = 36,
) -> list[Artist]:
    """Draw the 2D lidar sensing disc / wedge on a 3D axes at flight altitude."""
    arts: list[Artist] = []
    angles = np.linspace(yaw - fov / 2, yaw + fov / 2, n_segments + 1)
    verts = [[float(position[0]), float(position[1]), float(position[2])]]
    for a in angles:
        verts.append(
            [
                float(position[0] + max_range * np.cos(a)),
                float(position[1] + max_range * np.sin(a)),
                float(position[2]),
            ]
        )
    if abs(fov - 2 * np.pi) > 0.1:
        verts.append([float(position[0]), float(position[1]), float(position[2])])
    poly = Poly3DCollection([verts], alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.3)
    ax.add_collection3d(poly)
    arts.append(poly)
    return arts


def draw_lidar2d_rays_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    yaw: float,
    ranges: NDArray[np.floating],
    angles: NDArray[np.floating],
    max_range: float = 12.0,
    *,
    ray_color: str = C_LIDAR,
    hit_color: str = C_LIDAR_HIT,
    ray_alpha: float = 0.3,
    ray_lw: float = 0.5,
    hit_size: float = 6.0,
    every_n: int = 2,
) -> list[Artist]:
    """Draw 3D lidar beams from sensor origin to hit / max-range."""
    arts: list[Artist] = []
    segments = []
    hit_pts: list[list[float]] = []

    for i in range(0, len(ranges), every_n):
        a = yaw + angles[i]
        r = ranges[i]
        end = position + np.array([r * np.cos(a), r * np.sin(a), 0.0])
        segments.append([position.tolist(), end.tolist()])
        if r < max_range - 0.5:
            hit_pts.append(end.tolist())

    if segments:
        lc = Line3DCollection(segments, colors=ray_color, linewidths=ray_lw, alpha=ray_alpha)
        ax.add_collection3d(lc)
        arts.append(lc)

    if hit_pts:
        hp = np.array(hit_pts)
        sc = ax.scatter(hp[:, 0], hp[:, 1], hp[:, 2], c=hit_color, s=hit_size, zorder=8, alpha=0.6)
        arts.append(sc)

    return arts


# ── 3-D lidar FOV ────────────────────────────────────────────────────────────


def draw_lidar3d_fov_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    yaw: float,
    pitch: float = 0.0,
    h_fov: float = 2 * np.pi,
    v_fov: float = np.radians(30.0),
    max_range: float = 12.0,
    *,
    color: str = C_LIDAR3D,
    alpha: float = 0.04,
    n_segments: int = 24,
) -> list[Artist]:
    """Draw the 3D lidar FOV as a translucent cone / frustum."""
    arts: list[Artist] = []
    h_angles = np.linspace(yaw - h_fov / 2, yaw + h_fov / 2, n_segments + 1)
    v_half = v_fov / 2

    top_ring = []
    bot_ring = []
    for ha in h_angles:
        cos_top = np.cos(pitch + v_half)
        sin_top = np.sin(pitch + v_half)
        cos_bot = np.cos(pitch - v_half)
        sin_bot = np.sin(pitch - v_half)
        top_ring.append(
            [
                float(position[0] + max_range * cos_top * np.cos(ha)),
                float(position[1] + max_range * cos_top * np.sin(ha)),
                float(position[2] + max_range * sin_top),
            ]
        )
        bot_ring.append(
            [
                float(position[0] + max_range * cos_bot * np.cos(ha)),
                float(position[1] + max_range * cos_bot * np.sin(ha)),
                float(position[2] + max_range * sin_bot),
            ]
        )

    pos_list = [float(position[0]), float(position[1]), float(position[2])]
    for i in range(n_segments):
        tri_top = [pos_list, top_ring[i], top_ring[i + 1]]
        tri_bot = [pos_list, bot_ring[i], bot_ring[i + 1]]
        side = [top_ring[i], top_ring[i + 1], bot_ring[i + 1], bot_ring[i]]
        for verts in [tri_top, tri_bot, side]:
            poly = Poly3DCollection(
                [verts], alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.15
            )
            ax.add_collection3d(poly)
            arts.append(poly)
    return arts


def draw_lidar3d_points_3d(
    ax: Axes3D,
    point_cloud: NDArray[np.floating],
    *,
    cmap: str = "viridis",
    size: float = 4.0,
    alpha: float = 0.6,
    origin: NDArray[np.floating] | None = None,
) -> list[Artist]:
    """Render a 3D point cloud coloured by distance from *origin*."""
    arts: list[Artist] = []
    if len(point_cloud) == 0:
        return arts
    if origin is not None:
        dists = np.linalg.norm(point_cloud - origin, axis=1)
    else:
        dists = np.linalg.norm(point_cloud, axis=1)
    sc = ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c=dists,
        cmap=cmap,
        s=size,
        alpha=alpha,
        zorder=7,
    )
    arts.append(sc)
    return arts


# ── Camera frustum ────────────────────────────────────────────────────────────


def draw_camera_frustum_3d(
    ax: Axes3D,
    position: NDArray[np.floating],
    R: NDArray[np.floating],
    h_fov: float,
    v_fov: float,
    max_depth: float = 15.0,
    *,
    color: str = C_CAMERA,
    alpha: float = 0.08,
    edge_alpha: float = 0.4,
    mount_R: NDArray[np.floating] | None = None,
) -> list[Artist]:
    """Draw a camera view frustum as four semi-transparent triangular faces."""
    arts: list[Artist] = []
    d = max_depth
    half_h = d * np.tan(h_fov / 2)
    half_v = d * np.tan(v_fov / 2)

    corners_cam = np.array(
        [
            [-half_h, -half_v, d],
            [half_h, -half_v, d],
            [half_h, half_v, d],
            [-half_h, half_v, d],
        ]
    )
    R_full = R @ mount_R if mount_R is not None else R
    corners_world = (R_full @ corners_cam.T).T + position
    pos = [float(position[0]), float(position[1]), float(position[2])]

    for i in range(4):
        j = (i + 1) % 4
        tri = [pos, corners_world[i].tolist(), corners_world[j].tolist()]
        poly = Poly3DCollection(
            [tri], alpha=alpha, facecolor=color, edgecolor=color, linewidth=0.3
        )
        ax.add_collection3d(poly)
        arts.append(poly)

    far_face = Poly3DCollection(
        [corners_world.tolist()],
        alpha=alpha * 0.5,
        facecolor=color,
        edgecolor=color,
        linewidth=0.3,
    )
    ax.add_collection3d(far_face)
    arts.append(far_face)

    for c in corners_world:
        (edge,) = ax.plot(
            [position[0], c[0]],
            [position[1], c[1]],
            [position[2], c[2]],
            color=color,
            lw=0.5,
            alpha=edge_alpha,
        )
        arts.append(edge)

    return arts


def draw_camera_fov_top(
    ax: Any,
    position_xy: NDArray[np.floating],
    yaw: float,
    h_fov: float,
    max_depth: float = 15.0,
    *,
    color: str = C_CAMERA,
    alpha: float = 0.10,
    mount_yaw: float = 0.0,
) -> list[Artist]:
    """Draw camera FOV wedge on a top-down (XY) axes."""
    arts: list[Artist] = []
    total_yaw = yaw + mount_yaw
    start_deg = np.degrees(total_yaw - h_fov / 2)
    wedge = Wedge(
        position_xy,
        max_depth,
        start_deg,
        start_deg + np.degrees(h_fov),
        color=color,
        alpha=alpha,
        lw=0,
    )
    ax.add_patch(wedge)
    arts.append(wedge)
    wedge_edge = Wedge(
        position_xy,
        max_depth,
        start_deg,
        start_deg + np.degrees(h_fov),
        fill=False,
        edgecolor=color,
        alpha=alpha * 3,
        lw=0.6,
    )
    ax.add_patch(wedge_edge)
    arts.append(wedge_edge)
    return arts


def draw_camera_fov_side(
    ax: Any,
    position_xz: NDArray[np.floating],
    pitch: float,
    v_fov: float,
    max_depth: float = 15.0,
    *,
    color: str = C_CAMERA,
    alpha: float = 0.10,
    mount_pitch: float = 0.0,
) -> list[Artist]:
    """Draw camera FOV wedge on a side (XZ) axes."""
    arts: list[Artist] = []
    total_pitch = pitch + mount_pitch
    a1 = total_pitch - v_fov / 2
    a2 = total_pitch + v_fov / 2
    p1 = position_xz + max_depth * np.array([np.cos(a1), np.sin(a1)])
    p2 = position_xz + max_depth * np.array([np.cos(a2), np.sin(a2)])
    tri = Polygon(
        [position_xz, p1, p2],
        closed=True,
        color=color,
        alpha=alpha,
        lw=0,
    )
    ax.add_patch(tri)
    arts.append(tri)
    tri_edge = Polygon(
        [position_xz, p1, p2],
        closed=True,
        fill=False,
        edgecolor=color,
        alpha=alpha * 3,
        lw=0.6,
    )
    ax.add_patch(tri_edge)
    arts.append(tri_edge)
    return arts


# ── Lazy import helpers for patches that need plt ──────────────────────────


def plt_Circle(center, radius, **kwargs):
    """Thin wrapper so we don't import pyplot at module level."""
    import matplotlib.pyplot as plt

    return plt.Circle(center, radius, **kwargs)
