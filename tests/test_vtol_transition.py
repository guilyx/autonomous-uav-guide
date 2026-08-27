# Erwin Lejeune - 2026-02-17
"""Tests for the tilt-rotor VTOL model and its transition controller.

These pin down the three modelling errors the earlier tiltrotor had: an
angle of attack computed in the world frame (so pitch attitude did
nothing), wing lift gated on rotor tilt rather than airspeed, and lift that
never tilted with bank (so the aircraft could not turn).
"""

import numpy as np
import pytest

from flybots.control.vtol_controller import (
    VTOLCommand,
    VTOLController,
    VTOLGains,
    VTOLMode,
)
from flybots.vehicles.vtol import Tiltrotor, TiltrotorParams
from flybots.vehicles.vtol.tiltrotor import CRUISE_TILT, HOVER_TILT


def _cruising_state(airspeed: float = 20.0, pitch: float = 0.0, roll: float = 0.0):
    state = np.zeros(12)
    state[2] = 50.0
    state[3] = roll
    state[4] = pitch
    state[6] = airspeed
    return state


class TestAngleOfAttack:
    def test_alpha_responds_to_pitch_attitude(self):
        """The old model derived alpha from the world-frame flight path only."""
        vtol = Tiltrotor()
        for pitch_deg in (-15.0, -5.0, 5.0, 15.0):
            state = _cruising_state(pitch=np.radians(pitch_deg))
            vtol.reset(state=state)
            # FLU pitch is nose-down positive, so alpha is its negation.
            assert np.degrees(vtol.alpha) == pytest.approx(-pitch_deg, abs=0.5)

    def test_alpha_is_zero_in_level_flight(self):
        vtol = Tiltrotor()
        vtol.reset(state=_cruising_state())
        assert vtol.alpha == pytest.approx(0.0, abs=1e-6)

    def test_climbing_reduces_alpha_at_fixed_attitude(self):
        vtol = Tiltrotor()
        state = _cruising_state()
        state[8] = 5.0
        vtol.reset(state=state)
        assert vtol.alpha < 0.0


class TestWingLift:
    def test_lift_is_independent_of_rotor_tilt(self):
        """Wing lift comes from airflow, not from where the rotors point."""
        vtol = Tiltrotor()
        state = _cruising_state(pitch=np.radians(-6.0))
        lifts = []
        for tilt in (0.0, np.pi / 4, np.pi / 2):
            vtol.reset(state=state, tilt=tilt)
            vtol._dynamics(state, np.array([0.0, 0, 0, 0, tilt]))
            lifts.append(vtol.wing_lift)
        assert lifts[0] == pytest.approx(lifts[1])
        assert lifts[1] == pytest.approx(lifts[2])
        assert lifts[0] > 0.0

    def test_lift_grows_with_airspeed(self):
        vtol = Tiltrotor()
        lifts = []
        for airspeed in (0.0, 10.0, 20.0):
            state = _cruising_state(airspeed=airspeed, pitch=np.radians(-6.0))
            vtol.reset(state=state)
            vtol._dynamics(state, np.zeros(5))
            lifts.append(vtol.wing_lift)
        assert lifts[0] == pytest.approx(0.0, abs=1e-9)
        assert lifts[2] > lifts[1] > lifts[0]

    def test_lift_scales_roughly_quadratically(self):
        vtol = Tiltrotor()
        results = {}
        for airspeed in (10.0, 20.0):
            state = _cruising_state(airspeed=airspeed, pitch=np.radians(-6.0))
            vtol.reset(state=state)
            vtol._dynamics(state, np.zeros(5))
            results[airspeed] = vtol.wing_lift
        assert results[20.0] / results[10.0] == pytest.approx(4.0, rel=0.05)

    def test_lift_fraction_reports_weight_share(self):
        vtol = Tiltrotor()
        state = _cruising_state(airspeed=0.0)
        vtol.reset(state=state)
        vtol._dynamics(state, np.zeros(5))
        assert vtol.lift_fraction == pytest.approx(0.0)


class TestBankedTurn:
    def test_banking_produces_lateral_acceleration(self):
        """The old model's lift never rotated with roll, so it could not turn."""
        vtol = Tiltrotor()
        accelerations = []
        for roll_deg in (0.0, 20.0, 40.0):
            state = _cruising_state(pitch=np.radians(-8.0), roll=np.radians(roll_deg))
            vtol.reset(state=state, tilt=CRUISE_TILT)
            accelerations.append(vtol._dynamics(state, np.array([0.0, 0, 0, 0, CRUISE_TILT]))[7])
        assert accelerations[0] == pytest.approx(0.0, abs=1e-9)
        assert abs(accelerations[2]) > abs(accelerations[1]) > 0.5

    def test_bank_direction_sets_turn_direction(self):
        vtol = Tiltrotor()
        left = _cruising_state(pitch=np.radians(-8.0), roll=np.radians(-25.0))
        right = _cruising_state(pitch=np.radians(-8.0), roll=np.radians(25.0))
        vtol.reset(state=left, tilt=CRUISE_TILT)
        accel_left = vtol._dynamics(left, np.array([0.0, 0, 0, 0, CRUISE_TILT]))[7]
        vtol.reset(state=right, tilt=CRUISE_TILT)
        accel_right = vtol._dynamics(right, np.array([0.0, 0, 0, 0, CRUISE_TILT]))[7]
        assert accel_left * accel_right < 0.0


class TestTiltActuator:
    def test_tilt_is_rate_limited(self):
        params = TiltrotorParams(tilt_rate_limit=np.radians(10.0))
        vtol = Tiltrotor(params)
        vtol.reset(state=_cruising_state())
        vtol.step(np.array([0.0, 0, 0, 0, CRUISE_TILT]), 0.1)
        # One 0.1 s step can only move 1 degree, not the full 90.
        assert vtol.tilt == pytest.approx(np.radians(1.0), abs=1e-6)

    def test_tilt_is_clamped_to_max(self):
        vtol = Tiltrotor()
        vtol.reset(state=_cruising_state())
        for _ in range(2000):
            vtol.step(np.array([0.0, 0, 0, 0, 10.0]), 0.01)
        assert vtol.tilt == pytest.approx(vtol.vtol_params.max_tilt)

    def test_reset_restores_tilt(self):
        vtol = Tiltrotor()
        vtol.reset(state=_cruising_state(), tilt=CRUISE_TILT)
        assert vtol.tilt == pytest.approx(CRUISE_TILT)
        vtol.reset(state=_cruising_state())
        assert vtol.tilt == pytest.approx(HOVER_TILT)


class TestHover:
    def test_hover_thrust_holds_altitude(self):
        vtol = Tiltrotor()
        state = np.zeros(12)
        state[2] = 20.0
        vtol.reset(state=state)
        weight = vtol.vtol_params.mass * vtol.vtol_params.gravity
        for _ in range(500):
            vtol.step(np.array([weight, 0, 0, 0, HOVER_TILT]), 0.01)
        assert vtol.state[2] == pytest.approx(20.0, abs=0.05)

    def test_thrust_is_clamped_to_envelope(self):
        vtol = Tiltrotor()
        state = np.zeros(12)
        state[2] = 20.0
        vtol.reset(state=state)
        derivative = vtol._dynamics(state, np.array([1e9, 0, 0, 0, 0.0]))
        max_accel = vtol.vtol_params.max_thrust / vtol.vtol_params.mass
        assert derivative[8] <= max_accel - vtol.vtol_params.gravity + 1e-6


class TestParams:
    def test_derived_envelope_quantities(self):
        params = TiltrotorParams()
        assert params.max_thrust > params.mass * params.gravity
        assert params.max_torque > 0.0
        assert 0.0 < params.stall_airspeed < 30.0

    def test_legacy_aliases(self):
        params = TiltrotorParams()
        assert params.CL_alpha == params.coeffs.CLa
        assert params.CD0 == params.coeffs.CD0
        assert params.num_rotors == 4
        assert params.max_tilt == pytest.approx(np.pi / 2)


class TestTransitionController:
    @staticmethod
    def _fly(duration, cruise_window, dt=0.01, altitude=25.0):
        vtol = Tiltrotor()
        state = np.zeros(12)
        state[2] = altitude
        vtol.reset(state=state)
        pilot = VTOLController(vtol.vtol_params)
        command = VTOLCommand(altitude=altitude, cruise=False, cruise_airspeed=24.0)

        history = []
        for step in range(int(duration / dt)):
            t = step * dt
            command.cruise = cruise_window[0] <= t < cruise_window[1]
            vtol.step(pilot.compute(vtol.state, vtol.tilt, command, dt), dt)
            history.append((t, vtol.state[2], vtol.airspeed, vtol.tilt, pilot.mode))
        return vtol, pilot, history

    def test_reaches_wing_borne_cruise(self):
        vtol, pilot, _ = self._fly(90.0, (10.0, 90.0))
        assert pilot.mode is VTOLMode.CRUISE
        assert vtol.tilt == pytest.approx(CRUISE_TILT, abs=1e-3)
        # The wing must be carrying essentially the whole aircraft.
        assert vtol.lift_fraction > 0.9
        assert vtol.airspeed > vtol.vtol_params.stall_airspeed

    def test_holds_altitude_through_transition(self):
        _, _, history = self._fly(90.0, (10.0, 90.0))
        altitudes = np.array([h[1] for h in history])
        assert np.max(np.abs(altitudes - 25.0)) < 10.0
        # And settles tightly once established in cruise.
        settled = np.array([h[1] for h in history if h[0] > 45.0])
        assert np.max(np.abs(settled - 25.0)) < 1.0

    def test_back_transition_returns_to_hover(self):
        vtol, pilot, _ = self._fly(170.0, (10.0, 105.0))
        assert pilot.mode is VTOLMode.HOVER
        assert vtol.tilt == pytest.approx(HOVER_TILT, abs=1e-3)
        assert vtol.airspeed < 1.0
        assert vtol.state[2] == pytest.approx(25.0, abs=1.0)

    def test_mode_sequence_is_monotone(self):
        _, _, history = self._fly(170.0, (10.0, 105.0))
        modes = [h[4] for h in history]
        ordered = []
        for mode in modes:
            if not ordered or ordered[-1] is not mode:
                ordered.append(mode)
        assert ordered == [
            VTOLMode.HOVER,
            VTOLMode.TRANSITION,
            VTOLMode.CRUISE,
            VTOLMode.BACK_TRANSITION,
            VTOLMode.HOVER,
        ]

    def test_never_commands_stall_via_feedforward(self):
        params = TiltrotorParams()
        pilot = VTOLController(params)
        margin = pilot.gains.feedforward_stall_margin * params.coeffs.alpha_stall
        for airspeed in np.linspace(0.0, 40.0, 81):
            assert abs(pilot.level_flight_alpha(float(airspeed))) <= margin + 1e-9

    def test_level_flight_alpha_decreases_with_speed(self):
        pilot = VTOLController(TiltrotorParams())
        assert pilot.level_flight_alpha(30.0) < pilot.level_flight_alpha(18.0)

    def test_wing_authority_needs_both_speed_and_tilt(self):
        pilot = VTOLController(TiltrotorParams())
        fast_but_upright = pilot._wing_authority(30.0, HOVER_TILT)
        fast_and_tilted = pilot._wing_authority(30.0, CRUISE_TILT)
        assert fast_but_upright == pytest.approx(0.0)
        assert fast_and_tilted == pytest.approx(1.0)

    def test_torque_stays_within_actuator_limit(self):
        vtol = Tiltrotor()
        state = np.zeros(12)
        state[2] = 25.0
        vtol.reset(state=state)
        pilot = VTOLController(vtol.vtol_params)
        command = VTOLCommand(altitude=200.0, cruise=True, cruise_airspeed=24.0)
        limit = vtol.vtol_params.max_torque
        for _ in range(3000):
            control = pilot.compute(vtol.state, vtol.tilt, command, 0.01)
            assert np.all(np.abs(control[1:4]) <= limit + 1e-9)
            vtol.step(control, 0.01)

    def test_reset_returns_to_hover_mode(self):
        pilot = VTOLController(TiltrotorParams())
        pilot.mode = VTOLMode.CRUISE
        pilot.reset()
        assert pilot.mode is VTOLMode.HOVER

    def test_custom_gains_change_cruise_entry(self):
        params = TiltrotorParams()
        cautious = VTOLController(params, VTOLGains(stall_margin=1.6))
        eager = VTOLController(params, VTOLGains(stall_margin=1.1))
        assert cautious.cruise_entry_airspeed > eager.cruise_entry_airspeed
