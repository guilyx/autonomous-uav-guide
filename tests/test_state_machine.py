# Erwin Lejeune - 2026-02-15
"""Tests for the FlightMode state machine."""

import numpy as np

from uav_sim.control import FlightMode, StateManager
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


def _hover_sm() -> tuple[StateManager, Quadrotor]:
    """Helper: return a StateManager in HOVER at z=5."""
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    sm = StateManager(quad)
    sm.arm()
    sm.run_takeoff(altitude=5.0, dt=0.005, timeout=15.0)
    return sm, quad


class TestTransitions:
    def test_starts_disarmed(self):
        sm = StateManager(Quadrotor())
        assert sm.mode == FlightMode.DISARMED

    def test_arm(self):
        sm = StateManager(Quadrotor())
        assert sm.arm() is True
        assert sm.mode == FlightMode.ARMED

    def test_cannot_takeoff_when_disarmed(self):
        sm = StateManager(Quadrotor())
        assert sm.takeoff(5.0) is False
        assert sm.mode == FlightMode.DISARMED

    def test_arm_then_takeoff(self):
        sm = StateManager(Quadrotor())
        sm.arm()
        assert sm.takeoff(10.0) is True
        assert sm.mode == FlightMode.TAKEOFF

    def test_invalid_transition_returns_false(self):
        sm = StateManager(Quadrotor())
        sm.arm()
        assert sm.land() is False

    def test_land_from_hover(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 5.0]))
        sm = StateManager(quad)
        sm.arm()
        sm._mode = FlightMode.HOVER
        assert sm.land() is True
        assert sm.mode == FlightMode.LAND

    def test_hover_to_tracking(self):
        sm, _ = _hover_sm()
        assert sm.mode == FlightMode.HOVER
        wps = np.array([[5.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
        assert sm.tracking(wps) is True
        assert sm.mode == FlightMode.TRACKING

    def test_hover_to_position_hold(self):
        sm, _ = _hover_sm()
        assert sm.position_hold() is True
        assert sm.mode == FlightMode.POSITION_HOLD

    def test_hover_to_return_to_home(self):
        sm, _ = _hover_sm()
        assert sm.return_to_home() is True
        assert sm.mode == FlightMode.RETURN_TO_HOME

    def test_emergency_from_any(self):
        sm, _ = _hover_sm()
        assert sm.emergency() is True
        assert sm.mode == FlightMode.EMERGENCY

    def test_emergency_from_tracking(self):
        sm, _ = _hover_sm()
        wps = np.array([[5.0, 0.0, 5.0]])
        sm.tracking(wps)
        assert sm.emergency() is True
        assert sm.mode == FlightMode.EMERGENCY


class TestRunTakeoff:
    def test_reaches_altitude(self):
        sm, quad = _hover_sm()
        assert sm.mode == FlightMode.HOVER
        assert abs(quad.position[2] - 5.0) < 1.5

    def test_states_recorded(self):
        sm, _ = _hover_sm()
        assert len(sm.states) > 100


class TestFlyTo:
    def test_reaches_target(self):
        sm, quad = _hover_sm()
        sm.fly_to(np.array([5.0, 0.0, 5.0]), dt=0.005, threshold=2.0, timeout=20.0)
        dist = float(np.linalg.norm(quad.position - np.array([5.0, 0.0, 5.0])))
        assert dist < 3.0


class TestTrackPath:
    def test_follows_waypoints(self):
        sm, quad = _hover_sm()
        wps = np.array([[3.0, 0.0, 5.0], [6.0, 0.0, 5.0]])
        sm.track_path(wps, dt=0.005, timeout=40.0)
        assert sm.mode == FlightMode.HOVER
        dist = float(np.linalg.norm(quad.position[:2] - wps[-1, :2]))
        assert dist < 4.0

    def test_returns_to_hover_on_complete(self):
        sm, _ = _hover_sm()
        wps = np.array([[2.0, 0.0, 5.0]])
        sm.track_path(wps, dt=0.005, timeout=20.0)
        assert sm.mode == FlightMode.HOVER


class TestLand:
    def test_lands_to_disarmed(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 3.0]))
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))
        sm = StateManager(quad)
        sm.arm()
        sm._mode = FlightMode.HOVER
        sm._hold_pos = np.array([0.0, 0.0, 3.0])
        sm.run_land(dt=0.005, timeout=15.0)
        assert sm.mode == FlightMode.DISARMED
        assert quad.position[2] < 0.5
