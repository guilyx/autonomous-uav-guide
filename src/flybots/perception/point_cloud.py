# Erwin Lejeune - 2026-02-17
"""Point cloud utilities: conversion, voxel downsampling, ground removal."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ranges_to_point_cloud(
    position: NDArray[np.floating],
    yaw: float,
    ranges: NDArray[np.floating],
    angles: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convert 2-D lidar ranges + angles to an (N, 3) point cloud in world frame."""
    valid = np.isfinite(ranges) & (ranges > 0)
    r = ranges[valid]
    a = angles[valid] + yaw
    x = position[0] + r * np.cos(a)
    y = position[1] + r * np.sin(a)
    z = np.full_like(x, position[2])
    return np.column_stack([x, y, z])


def voxel_downsample(
    points: NDArray[np.floating],
    voxel_size: float = 0.2,
) -> NDArray[np.floating]:
    """Voxel-grid downsampling of a point cloud."""
    if len(points) == 0:
        return points
    quantised = np.floor(points / voxel_size).astype(int)
    _, idx = np.unique(quantised, axis=0, return_index=True)
    return points[np.sort(idx)]


def remove_ground(
    points: NDArray[np.floating],
    z_threshold: float = 0.3,
) -> NDArray[np.floating]:
    """Remove points below a height threshold (simple ground filter)."""
    return points[points[:, 2] > z_threshold]
