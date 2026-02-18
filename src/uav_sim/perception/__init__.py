# Erwin Lejeune - 2026-02-17
"""Perception algorithms: occupancy mapping, obstacle detection, point clouds."""

from uav_sim.perception.bbox_tracker import (
    Detection,
    SimulatedDetector,
    VisualServoConfig,
    VisualServoController,
)
from uav_sim.perception.obstacle_detection import RangeObstacleDetector
from uav_sim.perception.occupancy_mapping import OccupancyMapper
from uav_sim.perception.point_cloud import (
    ranges_to_point_cloud,
    remove_ground,
    voxel_downsample,
)

__all__ = [
    "Detection",
    "OccupancyMapper",
    "RangeObstacleDetector",
    "SimulatedDetector",
    "VisualServoConfig",
    "VisualServoController",
    "ranges_to_point_cloud",
    "remove_ground",
    "voxel_downsample",
]
