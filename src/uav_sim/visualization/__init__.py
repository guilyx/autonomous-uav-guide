# Erwin Lejeune - 2026-02-15
"""Visualisation toolkit for UAV simulations."""

from .animator import SimAnimator
from .costmap_viz import (
    create_four_panel_figure,
    draw_costmap_heatmap,
    draw_costmap_surface,
    draw_occupancy_overlay,
)
from .data_panel import (
    setup_attitude_panel,
    setup_error_panel,
    setup_estimation_panel,
    setup_position_panel,
    setup_thrust_panel,
    setup_velocity_panel,
    update_attitude_panel,
    update_error_panel,
    update_estimation_panel,
    update_position_panel,
    update_thrust_panel,
    update_velocity_panel,
)
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
    "create_four_panel_figure",
    "draw_camera_fov_side",
    "draw_camera_fov_top",
    "draw_camera_frustum_3d",
    "draw_costmap_heatmap",
    "draw_costmap_surface",
    "draw_fixed_wing_3d",
    "draw_hexarotor_3d",
    "draw_lidar2d_fov_3d",
    "draw_lidar2d_fov_top",
    "draw_lidar2d_rays_3d",
    "draw_lidar2d_rays_top",
    "draw_lidar3d_fov_3d",
    "draw_lidar3d_points_3d",
    "draw_occupancy_overlay",
    "draw_quadrotor_2d",
    "draw_quadrotor_3d",
    "plot_quadrotor_3d",
    "plot_state_history",
    "plot_trajectory_3d",
    "setup_attitude_panel",
    "setup_error_panel",
    "setup_estimation_panel",
    "setup_position_panel",
    "setup_thrust_panel",
    "setup_velocity_panel",
    "update_attitude_panel",
    "update_error_panel",
    "update_estimation_panel",
    "update_position_panel",
    "update_thrust_panel",
    "update_velocity_panel",
]
