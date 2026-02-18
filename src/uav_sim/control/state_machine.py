# Erwin Lejeune - 2026-02-15
"""PX4-inspired flight-mode state machine.

Manages mode transitions (DISARMED → ARMED → TAKEOFF → OFFBOARD / LOITER → LAND)
and dispatches setpoints to the underlying :class:`FlightController` each
simulation step.

Usage
-----
>>> sm = StateManager(quad)
>>> sm.arm()
>>> sm.takeoff(altitude=10.0)
>>> while not sm.is_mode(FlightMode.LOITER):
...     sm.step(dt)
>>> sm.offboard()
>>> sm.set_position_target(target)
>>> sm.step(dt)
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from uav_sim.control.flight_controller import FlightController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class FlightMode(Enum):
    DISARMED = auto()
    ARMED = auto()
    TAKEOFF = auto()
    LOITER = auto()
    OFFBOARD = auto()
    LAND = auto()


_VALID_TRANSITIONS: dict[FlightMode, set[FlightMode]] = {
    FlightMode.DISARMED: {FlightMode.ARMED},
    FlightMode.ARMED: {FlightMode.TAKEOFF, FlightMode.DISARMED},
    FlightMode.TAKEOFF: {FlightMode.LOITER, FlightMode.OFFBOARD, FlightMode.LAND},
    FlightMode.LOITER: {FlightMode.OFFBOARD, FlightMode.LAND, FlightMode.LOITER},
    FlightMode.OFFBOARD: {FlightMode.LOITER, FlightMode.LAND},
    FlightMode.LAND: {FlightMode.DISARMED},
}

_TAKEOFF_VEL = 1.5  # m/s vertical climb rate
_LAND_VEL = -0.8  # m/s descent rate
_LANDED_Z = 0.15  # m — consider landed when below this


class StateManager:
    """High-level flight mode manager wrapping a :class:`FlightController`.

    Parameters
    ----------
    quad : the quadrotor vehicle model.
    fc : optional pre-configured FlightController (created if None).
    """

    def __init__(
        self,
        quad: Quadrotor,
        fc: FlightController | None = None,
    ) -> None:
        self.quad = quad
        self.fc = fc or FlightController(mass=quad.params.mass, gravity=quad.params.gravity)
        self._mode = FlightMode.DISARMED
        self._takeoff_alt = 0.0
        self._loiter_pos: NDArray[np.floating] = np.zeros(3)
        self._states: list[NDArray[np.floating]] = []

    @property
    def mode(self) -> FlightMode:
        return self._mode

    def is_mode(self, m: FlightMode) -> bool:
        return self._mode == m

    @property
    def states(self) -> list[NDArray[np.floating]]:
        return self._states

    def _transition(self, target: FlightMode) -> bool:
        if target in _VALID_TRANSITIONS.get(self._mode, set()):
            self._mode = target
            return True
        return False

    def arm(self) -> bool:
        if not self._transition(FlightMode.ARMED):
            return False
        hover_f = self.quad.hover_wrench()[0] / 4.0
        for m in self.quad.motors:
            m.reset(m.thrust_to_omega(hover_f))
        self.fc.reset()
        return True

    def takeoff(self, altitude: float) -> bool:
        if not self._transition(FlightMode.TAKEOFF):
            return False
        self._takeoff_alt = altitude
        return True

    def land(self) -> bool:
        return self._transition(FlightMode.LAND)

    def offboard(self) -> bool:
        return self._transition(FlightMode.OFFBOARD)

    def loiter(self) -> bool:
        if not self._transition(FlightMode.LOITER):
            return False
        self._loiter_pos = self.quad.position.copy()
        return True

    def set_position_target(self, pos: NDArray[np.floating], yaw: float = 0.0) -> None:
        self.fc.set_position_target(pos, yaw)

    def set_velocity_target(self, vel: NDArray[np.floating], yaw: float = 0.0) -> None:
        self.fc.set_velocity_target(vel, yaw)

    def step(self, dt: float) -> NDArray[np.floating]:
        """Advance one simulation step. Returns the state after stepping."""
        if self._mode == FlightMode.DISARMED:
            wrench = np.zeros(4)
        elif self._mode == FlightMode.ARMED:
            wrench = self.quad.hover_wrench()
        elif self._mode == FlightMode.TAKEOFF:
            pos = self.quad.position
            target = pos.copy()
            target[2] = self._takeoff_alt
            self.fc.set_position_target(target)
            wrench = self.fc.compute(self.quad.state, dt)
            if abs(pos[2] - self._takeoff_alt) < 0.3 and abs(self.quad.velocity[2]) < 0.3:
                self._loiter_pos = self.quad.position.copy()
                self._mode = FlightMode.LOITER
        elif self._mode == FlightMode.LOITER:
            self.fc.set_position_target(self._loiter_pos)
            wrench = self.fc.compute(self.quad.state, dt)
        elif self._mode == FlightMode.OFFBOARD:
            wrench = self.fc.compute(self.quad.state, dt)
        elif self._mode == FlightMode.LAND:
            pos = self.quad.position
            target = pos.copy()
            target[2] = 0.0
            self.fc.set_position_target(target)
            wrench = self.fc.compute(self.quad.state, dt)
            if pos[2] < _LANDED_Z:
                self._mode = FlightMode.DISARMED
        else:
            wrench = np.zeros(4)

        self._states.append(self.quad.state.copy())
        self.quad.step(wrench, dt)
        return self.quad.state.copy()

    def run_takeoff(self, altitude: float, dt: float = 0.005, timeout: float = 8.0) -> None:
        """Convenience: run the full takeoff sequence until altitude is reached."""
        self.takeoff(altitude)
        max_steps = int(timeout / dt)
        for _ in range(max_steps):
            self.step(dt)
            if self._mode != FlightMode.TAKEOFF:
                break

    def run_land(self, dt: float = 0.005, timeout: float = 10.0) -> None:
        """Convenience: run the full landing sequence."""
        self.land()
        max_steps = int(timeout / dt)
        for _ in range(max_steps):
            self.step(dt)
            if self._mode == FlightMode.DISARMED:
                break

    def fly_to(
        self,
        target: NDArray[np.floating],
        dt: float = 0.005,
        threshold: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """Fly to a position target in OFFBOARD mode."""
        if self._mode == FlightMode.LOITER:
            self.offboard()
        self.set_position_target(target)
        max_steps = int(timeout / dt)
        for _ in range(max_steps):
            self.set_position_target(target)
            self.step(dt)
            if float(np.linalg.norm(self.quad.position - target)) < threshold:
                break
