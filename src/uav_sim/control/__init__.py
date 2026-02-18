# Erwin Lejeune - 2026-02-15
"""Layered control stack for multirotor UAVs.

Architecture (inner → outer):
  RateController → AttitudeController → VelocityController → PositionController

The :class:`FlightController` composes all four layers and routes commands
based on the active :class:`ControlMode`.
"""

from uav_sim.control.attitude_controller import AttitudeController
from uav_sim.control.flight_controller import ControlMode, FlightController
from uav_sim.control.position_controller import PositionController
from uav_sim.control.rate_controller import RateController
from uav_sim.control.state_machine import FlightMode, StateManager
from uav_sim.control.velocity_controller import VelocityController

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
