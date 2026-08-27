# Erwin Lejeune - 2026-02-17
"""Vehicle models: multirotor, VTOL, fixed-wing."""

from flybots.vehicles.base import UAVBase, UAVParams
from flybots.vehicles.components.allocation import (
    ControlAllocation,
    Rotor,
    coaxial_layout,
    h_layout,
    plus_layout,
    radial_layout,
    x_layout,
)
from flybots.vehicles.footprint import (
    BaseFootprint,
    CircularFootprint,
    RectangularFootprint,
    swarm_convex_hull,
)
from flybots.vehicles.multirotor import (
    Multirotor,
    MultirotorParams,
    Quadrotor,
    QuadrotorParams,
)
from flybots.vehicles.presets import (
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
