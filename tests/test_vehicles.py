# Erwin Lejeune - 2026-02-17
"""Tests for the vehicle hierarchy: UAVBase, Quadrotor, FixedWing, Tiltrotor."""

import numpy as np
import pytest

from uav_sim.vehicles.base import UAVBase, UAVParams
from uav_sim.vehicles.fixed_wing import FixedWing, FixedWingParams
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.vehicles.vtol import Tiltrotor, TiltrotorParams


class TestUAVBase:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            UAVBase()

    def test_params_defaults(self):
        p = UAVParams()
        assert p.mass == 1.0
        assert p.gravity == 9.81


class TestQuadrotor:
    def test_state_size(self):
        q = Quadrotor()
        assert q.STATE_SIZE == 12
        assert q.state.shape == (12,)

    def test_hover_thrust(self):
        q = Quadrotor()
        q.reset(position=np.array([0.0, 0.0, 1.0]))
        hover_thrust = q.params.mass * q.params.gravity
        u = np.array([hover_thrust, 0, 0, 0])
        hover_f = hover_thrust / 4.0
        for m in q.motors:
            m.reset(m.thrust_to_omega(hover_f))
        z0 = q.state[2]
        for _ in range(100):
            q.step(u, 0.01)
        assert abs(q.state[2] - z0) < 0.5


class TestFixedWing:
    def test_state_dim(self):
        fw = FixedWing()
        assert fw.state_dim == 12
        assert fw.control_dim == 4

    def test_forward_flight(self):
        fw = FixedWing()
        fw.reset()
        state = np.zeros(12)
        state[6] = 15.0  # forward velocity u
        state[2] = 100.0  # altitude
        fw.reset(state=state)
        control = np.array([0.0, 0.0, 0.0, 0.5])  # throttle
        for _ in range(100):
            fw.step(control, 0.01)
        assert fw.state[0] > 0  # moved forward

    def test_params(self):
        p = FixedWingParams()
        assert p.wing_area > 0
        assert p.mass > 0


class TestTiltrotor:
    def test_state_dim(self):
        tr = Tiltrotor()
        assert tr.state_dim == 12
        assert tr.control_dim == 5

    def test_hover(self):
        tr = Tiltrotor()
        state = np.zeros(12)
        state[2] = 5.0
        tr.reset(state=state)
        T = tr.vtol_params.mass * tr.vtol_params.gravity
        control = np.array([T, 0, 0, 0, 0])  # hover, tilt=0
        z0 = tr.state[2]
        for _ in range(100):
            tr.step(control, 0.01)
        assert abs(tr.state[2] - z0) < 0.5

    def test_params(self):
        p = TiltrotorParams()
        assert p.num_rotors == 4
        assert p.max_tilt == pytest.approx(np.pi / 2)
