# Erwin Lejeune - 2026-02-15
"""Layered control stack for multirotor UAVs.

Architecture (inner → outer):
  RateController → AttitudeController → VelocityController → PositionController

The :class:`FlightController` composes all four layers and routes commands
based on the active :class:`ControlMode`.
"""

from flybots.control.attitude_controller import AttitudeController
from flybots.control.flight_controller import ControlMode, FlightController
from flybots.control.position_controller import PositionController
from flybots.control.rate_controller import RateController
from flybots.control.state_machine import FlightMode, StateManager
from flybots.control.velocity_controller import VelocityController

__all__ = [
    "AttitudeController",
    "ControlMode",
    "FlightController",
    "FlightMode",
    "PositionController",
    "RateController",
    "StateManager",
    "VelocityController",
]
