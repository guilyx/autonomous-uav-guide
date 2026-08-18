# Erwin Lejeune - 2026-02-17
"""Vehicle models: multirotor, VTOL, fixed-wing."""

from uav_sim.vehicles.base import UAVBase, UAVParams
from uav_sim.vehicles.components.allocation import (
    ControlAllocation,
    Rotor,
    coaxial_layout,
    h_layout,
    plus_layout,
    radial_layout,
    x_layout,
)
from uav_sim.vehicles.footprint import (
    BaseFootprint,
    CircularFootprint,
    RectangularFootprint,
    swarm_convex_hull,
)
from uav_sim.vehicles.multirotor import (
    Multirotor,
    MultirotorParams,
    Quadrotor,
    QuadrotorParams,
)
from uav_sim.vehicles.presets import (
    VehiclePreset,
    create_multirotor,
    create_quadrotor,
    get_params,
)

__all__ = [
    "BaseFootprint",
    "CircularFootprint",
    "ControlAllocation",
    "Multirotor",
    "MultirotorParams",
    "Quadrotor",
    "QuadrotorParams",
    "RectangularFootprint",
    "Rotor",
    "UAVBase",
    "UAVParams",
    "VehiclePreset",
    "coaxial_layout",
    "create_multirotor",
    "create_quadrotor",
    "get_params",
    "h_layout",
    "plus_layout",
    "radial_layout",
    "swarm_convex_hull",
    "x_layout",
]
