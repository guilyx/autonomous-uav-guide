# Erwin Lejeune - 2026-02-18
"""Visualisation toolkit for UAV simulations."""

from .animator import SimAnimator
from .plotting import plot_quadrotor_3d, plot_state_history, plot_trajectory_3d
from .sensor_viz import (
    draw_camera_fov_side,
    draw_camera_fov_top,
    draw_camera_frustum_3d,
    draw_lidar2d_fov_3d,
    draw_lidar2d_fov_top,
    draw_lidar2d_rays_3d,
    draw_lidar2d_rays_top,
    draw_lidar3d_fov_3d,
    draw_lidar3d_points_3d,
)
from .three_panel import ThreePanelViz
from .vehicle_artists import (
    clear_vehicle_artists,
    draw_fixed_wing_3d,
    draw_hexarotor_3d,
    draw_quadrotor_2d,
    draw_quadrotor_3d,
)

__all__ = [
    "SimAnimator",
    "ThreePanelViz",
    "clear_vehicle_artists",
    "draw_camera_fov_side",
    "draw_camera_fov_top",
    "draw_camera_frustum_3d",
    "draw_fixed_wing_3d",
    "draw_hexarotor_3d",
    "draw_lidar2d_fov_3d",
    "draw_lidar2d_fov_top",
    "draw_lidar2d_rays_3d",
    "draw_lidar2d_rays_top",
    "draw_lidar3d_fov_3d",
    "draw_lidar3d_points_3d",
    "draw_quadrotor_2d",
    "draw_quadrotor_3d",
    "plot_quadrotor_3d",
    "plot_state_history",
    "plot_trajectory_3d",
]
