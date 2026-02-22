# Erwin Lejeune - 2026-02-15
"""Coordinate frame utilities for UAV systems.

Provides conversions between world (ENU), body (FCU), and sensor frames
for 3-D navigation and perception.  Inspired by
``simple_autonomous_car.frames``, extended to three dimensions.
"""

from uav_sim.frames.transforms import (
    FrameID,
    batch_body_to_world,
    batch_sensor_to_world,
    batch_world_to_body,
    body_to_sensor,
    body_to_world,
    euler_to_rotation,
    sensor_to_body,
    sensor_to_world,
    world_to_body,
    world_to_sensor,
)

__all__ = [
    "FrameID",
    "batch_body_to_world",
    "batch_sensor_to_world",
    "batch_world_to_body",
    "body_to_sensor",
    "body_to_world",
    "euler_to_rotation",
    "sensor_to_body",
    "sensor_to_world",
    "world_to_body",
    "world_to_sensor",
]
