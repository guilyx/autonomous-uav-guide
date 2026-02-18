# Erwin Lejeune - 2026-02-17
"""Path tracking controllers: PID, LQR, Geometric SO(3), Pure-Pursuit 3D."""

from uav_sim.path_tracking.geometric_controller import GeometricController
from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D

__all__ = [
    "CascadedPIDController",
    "GeometricController",
    "LQRController",
    "PurePursuit3D",
    "smooth_path_3d",
]
