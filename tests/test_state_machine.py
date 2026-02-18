# Erwin Lejeune - 2026-02-15
"""Tests for the FlightMode state machine."""

import numpy as np

from uav_sim.control import FlightMode, StateManager
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


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

    def test_land_from_loiter(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 5.0]))
        sm = StateManager(quad)
        sm.arm()
        sm._mode = FlightMode.LOITER
        assert sm.land() is True
        assert sm.mode == FlightMode.LAND


class TestRunTakeoff:
    def test_reaches_altitude(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.0]))
        sm = StateManager(quad)
        sm.arm()
        sm.run_takeoff(altitude=5.0, dt=0.005, timeout=15.0)
        assert sm.mode == FlightMode.LOITER
        assert abs(quad.position[2] - 5.0) < 1.5

    def test_states_recorded(self):
        quad = Quadrotor()
        sm = StateManager(quad)
        sm.arm()
        sm.run_takeoff(altitude=5.0, dt=0.005, timeout=10.0)
        assert len(sm.states) > 100


class TestFlyTo:
    def test_reaches_target(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.0]))
        sm = StateManager(quad)
        sm.arm()
        sm.run_takeoff(altitude=5.0, dt=0.005, timeout=12.0)
        sm.fly_to(np.array([5.0, 0.0, 5.0]), dt=0.005, threshold=2.0, timeout=20.0)
        dist = float(np.linalg.norm(quad.position - np.array([5.0, 0.0, 5.0])))
        assert dist < 3.0


class TestLand:
    def test_lands_to_disarmed(self):
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 3.0]))
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))
        sm = StateManager(quad)
        sm.arm()
        sm._mode = FlightMode.LOITER
        sm._loiter_pos = np.array([0.0, 0.0, 3.0])
        sm.run_land(dt=0.005, timeout=15.0)
        assert sm.mode == FlightMode.DISARMED
        assert quad.position[2] < 0.5
