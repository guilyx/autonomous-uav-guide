# Erwin Lejeune - 2026-02-17
"""Multirotor vehicle models."""

from flybots.vehicles.components.allocation import (
    ControlAllocation,
    Rotor,
    coaxial_layout,
    h_layout,
    plus_layout,
    radial_layout,
    x_layout,
)
from flybots.vehicles.multirotor.multirotor import Multirotor, MultirotorParams
from flybots.vehicles.multirotor.quadrotor import Quadrotor, QuadrotorParams

__all__ = [
    "ControlAllocation",
    "Multirotor",
    "MultirotorParams",
    "Quadrotor",
    "QuadrotorParams",
    "Rotor",
    "coaxial_layout",
    "h_layout",
    "plus_layout",
    "radial_layout",
    "x_layout",
]
