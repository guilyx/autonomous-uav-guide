# Erwin Lejeune - 2026-02-15
"""Composite flight controller that stacks the four control layers.

Supports multiple :class:`ControlMode` values so callers can inject
commands at any abstraction level (rates, attitude, velocity, or position).
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from uav_sim.control.attitude_controller import AttitudeController
from uav_sim.control.position_controller import PositionController
from uav_sim.control.velocity_controller import VelocityController


class ControlMode(Enum):
    """Which level of the stack is receiving external setpoints."""

    POSITION = auto()
    VELOCITY = auto()
    ATTITUDE = auto()
    RATE = auto()


class FlightController:
    """Full control stack: position → velocity → attitude → rate → wrench.

    Parameters
    ----------
    mode : initial control mode.
    mass : vehicle mass [kg] (used for hover thrust default).
    gravity : gravitational acceleration [m/s^2].
    """

    def __init__(
        self,
        mode: ControlMode = ControlMode.POSITION,
        mass: float = 1.5,
        gravity: float = 9.81,
    ) -> None:
        self.mode = mode
        self.mass = mass
        self.gravity = gravity

        self.pos_ctrl = PositionController()
        self.vel_ctrl = VelocityController()
        self.att_ctrl = AttitudeController()

        self._target_pos = np.zeros(3)
        self._target_vel = np.zeros(3)
        self._target_euler = np.zeros(3)
        self._target_thrust = mass * gravity
        self._target_yaw = 0.0

    def reset(self) -> None:
        self.pos_ctrl.reset()
        self.vel_ctrl.reset()
        self.att_ctrl.reset()
        self._target_pos = np.zeros(3)
        self._target_vel = np.zeros(3)
        self._target_euler = np.zeros(3)
        self._target_thrust = self.mass * self.gravity
        self._target_yaw = 0.0

    def set_position_target(self, pos: NDArray[np.floating], yaw: float = 0.0) -> None:
        self.mode = ControlMode.POSITION
        self._target_pos = np.asarray(pos, dtype=np.float64)
        self._target_yaw = yaw

    def set_velocity_target(self, vel: NDArray[np.floating], yaw: float = 0.0) -> None:
        self.mode = ControlMode.VELOCITY
        self._target_vel = np.asarray(vel, dtype=np.float64)
        self._target_yaw = yaw

    def set_attitude_target(self, euler: NDArray[np.floating], thrust: float) -> None:
        self.mode = ControlMode.ATTITUDE
        self._target_euler = np.asarray(euler, dtype=np.float64)
        self._target_thrust = thrust

    def compute(self, state: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Run the full control stack and return ``[T, tau_x, tau_y, tau_z]``."""
        pos = state[:3]
        euler = state[3:6]
        vel = state[6:9]
        omega = state[9:12]

        if self.mode == ControlMode.POSITION:
            des_vel = self.pos_ctrl.compute(pos, self._target_pos, dt, velocity=vel)
            des_euler, thrust = self.vel_ctrl.compute(vel, self._target_yaw, des_vel, dt)
            return self.att_ctrl.compute(euler, omega, des_euler, thrust, dt)

        if self.mode == ControlMode.VELOCITY:
            des_euler, thrust = self.vel_ctrl.compute(vel, self._target_yaw, self._target_vel, dt)
            return self.att_ctrl.compute(euler, omega, des_euler, thrust, dt)

        if self.mode == ControlMode.ATTITUDE:
            return self.att_ctrl.compute(euler, omega, self._target_euler, self._target_thrust, dt)

        # RATE mode — direct torque pass-through (not yet fully wired)
        return np.array([self._target_thrust, 0.0, 0.0, 0.0])
