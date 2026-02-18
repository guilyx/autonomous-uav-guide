# Erwin Lejeune - 2026-02-15
"""PX4-inspired flight-mode state machine.

Manages mode transitions and dispatches setpoints to the underlying
:class:`FlightController` each simulation step.

Mode graph (PX4 conventions)::

    DISARMED -> ARMED -> TAKEOFF -> HOVER
    HOVER -> {OFFBOARD, TRACKING, POSITION_HOLD, RETURN_TO_HOME, LAND}
    OFFBOARD -> HOVER
    TRACKING -> HOVER
    POSITION_HOLD -> HOVER
    RETURN_TO_HOME -> LAND
    LAND -> DISARMED
    ANY -> EMERGENCY -> DISARMED

Usage
-----
>>> sm = StateManager(quad)
>>> sm.arm()
>>> sm.run_takeoff(altitude=10.0)
>>> sm.track_path(waypoints, dt=0.005)
>>> sm.run_land()
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
    HOVER = auto()
    OFFBOARD = auto()
    POSITION_HOLD = auto()
    TRACKING = auto()
    RETURN_TO_HOME = auto()
    LAND = auto()
    EMERGENCY = auto()

    # Legacy alias kept for backward compat with existing simulations
    LOITER = HOVER


_VALID_TRANSITIONS: dict[FlightMode, set[FlightMode]] = {
    FlightMode.DISARMED: {FlightMode.ARMED},
    FlightMode.ARMED: {FlightMode.TAKEOFF, FlightMode.DISARMED},
    FlightMode.TAKEOFF: {FlightMode.HOVER, FlightMode.LAND},
    FlightMode.HOVER: {
        FlightMode.OFFBOARD,
        FlightMode.TRACKING,
        FlightMode.POSITION_HOLD,
        FlightMode.RETURN_TO_HOME,
        FlightMode.LAND,
    },
    FlightMode.OFFBOARD: {FlightMode.HOVER, FlightMode.LAND},
    FlightMode.TRACKING: {FlightMode.HOVER, FlightMode.LAND},
    FlightMode.POSITION_HOLD: {FlightMode.HOVER, FlightMode.LAND},
    FlightMode.RETURN_TO_HOME: {FlightMode.LAND},
    FlightMode.LAND: {FlightMode.DISARMED},
    FlightMode.EMERGENCY: {FlightMode.DISARMED},
}

_LANDED_Z = 0.15


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
        self._home: NDArray[np.floating] = np.zeros(3)
        self._hold_pos: NDArray[np.floating] = np.zeros(3)
        self._track_wps: NDArray[np.floating] | None = None
        self._track_idx: int = 0
        self._states: list[NDArray[np.floating]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> FlightMode:
        return self._mode

    def is_mode(self, m: FlightMode) -> bool:
        return self._mode == m

    @property
    def states(self) -> list[NDArray[np.floating]]:
        return self._states

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def _transition(self, target: FlightMode) -> bool:
        if target == FlightMode.EMERGENCY:
            self._mode = FlightMode.EMERGENCY
            return True
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
        self._home = self.quad.position.copy()
        return True

    def takeoff(self, altitude: float) -> bool:
        if not self._transition(FlightMode.TAKEOFF):
            return False
        self._takeoff_alt = altitude
        return True

    def hover(self) -> bool:
        if not self._transition(FlightMode.HOVER):
            return False
        self._hold_pos = self.quad.position.copy()
        return True

    def offboard(self) -> bool:
        return self._transition(FlightMode.OFFBOARD)

    def position_hold(self) -> bool:
        if not self._transition(FlightMode.POSITION_HOLD):
            return False
        self._hold_pos = self.quad.position.copy()
        return True

    def tracking(self, waypoints: NDArray[np.floating]) -> bool:
        """Enter TRACKING mode to follow a sequence of waypoints."""
        if not self._transition(FlightMode.TRACKING):
            return False
        self._track_wps = np.asarray(waypoints, dtype=np.float64)
        self._track_idx = 0
        return True

    def return_to_home(self) -> bool:
        return self._transition(FlightMode.RETURN_TO_HOME)

    def land(self) -> bool:
        return self._transition(FlightMode.LAND)

    def emergency(self) -> bool:
        return self._transition(FlightMode.EMERGENCY)

    # Legacy alias
    def loiter(self) -> bool:
        return self.hover()

    # ------------------------------------------------------------------
    # Setpoint passthrough for OFFBOARD mode
    # ------------------------------------------------------------------

    def set_position_target(self, pos: NDArray[np.floating], yaw: float = 0.0) -> None:
        self.fc.set_position_target(pos, yaw)

    def set_velocity_target(self, vel: NDArray[np.floating], yaw: float = 0.0) -> None:
        self.fc.set_velocity_target(vel, yaw)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float) -> NDArray[np.floating]:
        """Advance one simulation step. Returns the state after stepping."""
        wrench = self._compute_wrench(dt)
        self._states.append(self.quad.state.copy())
        self.quad.step(wrench, dt)
        return self.quad.state.copy()

    def _compute_wrench(self, dt: float) -> NDArray[np.floating]:
        mode = self._mode

        if mode == FlightMode.DISARMED or mode == FlightMode.EMERGENCY:
            return np.zeros(4)

        if mode == FlightMode.ARMED:
            return self.quad.hover_wrench()

        if mode == FlightMode.TAKEOFF:
            target = self.quad.position.copy()
            target[2] = self._takeoff_alt
            self.fc.set_position_target(target)
            w = self.fc.compute(self.quad.state, dt)
            if (
                abs(self.quad.position[2] - self._takeoff_alt) < 0.3
                and abs(self.quad.velocity[2]) < 0.3
            ):
                self._hold_pos = self.quad.position.copy()
                self._mode = FlightMode.HOVER
            return w

        if mode == FlightMode.HOVER or mode == FlightMode.POSITION_HOLD:
            self.fc.set_position_target(self._hold_pos)
            return self.fc.compute(self.quad.state, dt)

        if mode == FlightMode.OFFBOARD:
            return self.fc.compute(self.quad.state, dt)

        if mode == FlightMode.TRACKING:
            return self._tracking_step(dt)

        if mode == FlightMode.RETURN_TO_HOME:
            self.fc.set_position_target(self._home)
            w = self.fc.compute(self.quad.state, dt)
            if float(np.linalg.norm(self.quad.position - self._home)) < 1.0:
                self._mode = FlightMode.LAND
            return w

        if mode == FlightMode.LAND:
            target = self.quad.position.copy()
            target[2] = 0.0
            self.fc.set_position_target(target)
            w = self.fc.compute(self.quad.state, dt)
            if self.quad.position[2] < _LANDED_Z:
                self._mode = FlightMode.DISARMED
            return w

        return np.zeros(4)

    def _tracking_step(self, dt: float) -> NDArray[np.floating]:
        """Pure-pursuit style waypoint tracking in TRACKING mode."""
        if self._track_wps is None or self._track_idx >= len(self._track_wps):
            self._hold_pos = self.quad.position.copy()
            self._mode = FlightMode.HOVER
            return self.fc.compute(self.quad.state, dt)

        target = self._track_wps[self._track_idx]
        self.fc.set_position_target(target)
        w = self.fc.compute(self.quad.state, dt)

        if float(np.linalg.norm(self.quad.position - target)) < 1.5:
            self._track_idx += 1
            if self._track_idx >= len(self._track_wps):
                self._hold_pos = self.quad.position.copy()
                self._mode = FlightMode.HOVER
        return w

    # ------------------------------------------------------------------
    # Convenience runners
    # ------------------------------------------------------------------

    def run_takeoff(self, altitude: float, dt: float = 0.005, timeout: float = 8.0) -> None:
        """Run the full takeoff sequence until altitude is reached."""
        self.takeoff(altitude)
        for _ in range(int(timeout / dt)):
            self.step(dt)
            if self._mode != FlightMode.TAKEOFF:
                break

    def run_land(self, dt: float = 0.005, timeout: float = 10.0) -> None:
        """Run the full landing sequence."""
        self.land()
        for _ in range(int(timeout / dt)):
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
        if self._mode == FlightMode.HOVER:
            self.offboard()
        self.set_position_target(target)
        for _ in range(int(timeout / dt)):
            self.set_position_target(target)
            self.step(dt)
            if float(np.linalg.norm(self.quad.position - target)) < threshold:
                break

    def track_path(
        self,
        waypoints: NDArray[np.floating],
        dt: float = 0.005,
        timeout: float = 120.0,
    ) -> None:
        """Follow a sequence of waypoints to completion.

        Enters TRACKING mode from HOVER and runs until all waypoints
        are reached (auto-returns to HOVER) or *timeout* is exceeded.
        """
        if self._mode == FlightMode.HOVER:
            self.tracking(waypoints)
        elif self._mode == FlightMode.OFFBOARD:
            self.hover()
            self.tracking(waypoints)
        else:
            return
        for _ in range(int(timeout / dt)):
            self.step(dt)
            if self._mode != FlightMode.TRACKING:
                break
