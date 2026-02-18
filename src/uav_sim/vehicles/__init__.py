# Erwin Lejeune - 2026-02-17
"""Vehicle models: multirotor, VTOL, fixed-wing."""

from uav_sim.vehicles.base import UAVBase, UAVParams
from uav_sim.vehicles.footprint import (
    BaseFootprint,
    CircularFootprint,
    RectangularFootprint,
    swarm_convex_hull,
)
from uav_sim.vehicles.presets import VehiclePreset, create_quadrotor, get_params

__all__ = [
    "BaseFootprint",
    "CircularFootprint",
    "RectangularFootprint",
    "UAVBase",
    "UAVParams",
    "VehiclePreset",
    "create_quadrotor",
    "get_params",
    "swarm_convex_hull",
]
