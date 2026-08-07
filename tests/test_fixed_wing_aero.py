# Erwin Lejeune - 2026-02-17
"""Tests for the fixed-wing aerodynamic model, trim solver and autopilot.

These lock in the physics that the earlier simplified model got wrong:
frame consistency, live stability derivatives, stall behaviour, and the
existence of a genuine flight equilibrium.
"""

import numpy as np
import pytest

from uav_sim.control.fixed_wing_autopilot import (
    AutopilotCommand,
    AutopilotGains,
    FixedWingAutopilot,
)
from uav_sim.vehicles.fixed_wing import (
    AeroCoefficients,
    FixedWing,
    FixedWingParams,
    FixedWingPreset,
    TrimError,
    compute_trim,
    create_fixed_wing,
    get_fixed_wing_params,
)
from uav_sim.vehicles.fixed_wing.aerodynamics import (
    PropulsionParams,
    airframe_wrench,
    drag_coefficient,
    lift_coefficient,
    propeller_thrust,
)

ALL_PRESETS = [p for p in FixedWingPreset if p is not FixedWingPreset.CUSTOM]


def _trim_controls(aircraft: FixedWing, **kwargs) -> np.ndarray:
    return aircraft.reset_trimmed(**kwargs)


class TestFrameConventions:
    def test_altitude_increases_upward(self):
        """z is ENU altitude, not NED depth.

        The previous model integrated ``z`` downward while every other
        vehicle in the library treated it as altitude.
        """
        aircraft = FixedWing()
        controls = _trim_controls(aircraft, airspeed=35.0, altitude=200.0, climb_rate=3.0)
        for _ in range(500):
            aircraft.step(controls, 0.01)
        assert aircraft.state[2] > 200.0

    def test_gliding_descends(self):
        aircraft = FixedWing()
        state = np.zeros(12)
        state[2] = 500.0
        state[6] = 30.0
        aircraft.reset(state=state)
        for _ in range(500):
            aircraft.step(np.zeros(4), 0.01)
        assert aircraft.state[2] < 500.0

    def test_pitch_up_property_negates_flu_theta(self):
        aircraft = FixedWing()
        state = np.zeros(12)
        state[4] = 0.2
        aircraft.reset(state=state)
        assert aircraft.pitch_up == pytest.approx(-0.2)


class TestStabilityDerivatives:
    """Every derivative must actually influence the dynamics."""

    def test_static_pitch_stability_is_restoring(self):
        """Cma < 0 must produce a moment opposing an angle-of-attack change."""
        aircraft = FixedWing()
        controls = _trim_controls(aircraft, airspeed=35.0)
        base = aircraft.state.copy()
        alpha0 = aircraft.alpha

        moments = {}
        for delta in (-0.1, 0.1):
            state = base.copy()
            alpha = alpha0 + delta
            state[6] = 35.0 * np.cos(alpha)
            state[8] = -35.0 * np.sin(alpha)
            moments[delta] = aircraft._dynamics(state, controls)[10]

        # FLU pitch is nose-down positive, so a nose-up disturbance
        # (positive alpha) must generate a positive (nose-down) rate.
        assert moments[0.1] > 0.0
        assert moments[-0.1] < 0.0

    def test_pitch_damping_settles_short_period(self):
        """Cmq < 0 must damp the pitch oscillation."""
        aircraft = FixedWing()
        controls = _trim_controls(aircraft, airspeed=35.0, altitude=500.0)
        state = aircraft.state.copy()
        state[10] = 1.0
        aircraft.reset(state=state)

        pitch = []
        for _ in range(4000):
            aircraft.step(controls, 0.002)
            pitch.append(aircraft.pitch_up)
        pitch = np.array(pitch)
        assert np.ptp(pitch[-1000:]) < 0.25 * np.ptp(pitch[:1000])

    def test_weathercock_yaws_into_the_wind(self):
        """Cnb > 0 must turn the nose toward the relative wind."""
        aircraft = FixedWing()
        controls = _trim_controls(aircraft, airspeed=35.0)
        state = aircraft.state.copy()
        state[7] = 5.0
        derivative = aircraft._dynamics(state, controls)
        assert derivative[11] * state[7] > 0.0

    def test_sideslip_generates_side_force(self):
        """CYb must produce a lateral force; the old model had none."""
        aircraft = FixedWing()
        controls = _trim_controls(aircraft, airspeed=35.0)
        state = aircraft.state.copy()
        state[7] = 5.0
        assert abs(aircraft._dynamics(state, controls)[7]) > 1.0

    def test_dihedral_rolls_out_of_sideslip(self):
        aircraft = FixedWing()
        controls = _trim_controls(aircraft, airspeed=35.0)
        state = aircraft.state.copy()
        state[7] = 5.0
        assert abs(aircraft._dynamics(state, controls)[9]) > 1e-3

    @pytest.mark.parametrize("name", ["Cm0", "Cma", "e_oswald"])
    def test_formerly_dead_parameters_change_the_dynamics(self, name):
        """These were declared but never read by the old model."""
        base = FixedWing()
        controls = _trim_controls(base, airspeed=35.0)
        state = base.state.copy()
        state[8] = -2.0

        perturbed_coeffs = AeroCoefficients()
        setattr(perturbed_coeffs, name, getattr(perturbed_coeffs, name) * 2.0 + 0.05)
        other = FixedWing(FixedWingParams(coeffs=perturbed_coeffs))

        assert not np.allclose(base._dynamics(state, controls), other._dynamics(state, controls))


class TestStallModel:
    def test_lift_peaks_then_drops(self):
        coeffs = AeroCoefficients()
        alphas = np.radians(np.arange(0, 70, 1.0))
        cl = np.array([lift_coefficient(float(a), coeffs) for a in alphas])
        peak = int(np.argmax(cl))
        assert 0 < peak < len(cl) - 1
        assert cl[-1] < cl[peak]

    def test_lift_is_finite_at_extreme_incidence(self):
        coeffs = AeroCoefficients()
        for alpha in np.linspace(-np.pi, np.pi, 361):
            assert np.isfinite(lift_coefficient(float(alpha), coeffs))

    def test_induced_drag_uses_aspect_ratio(self):
        coeffs = AeroCoefficients()
        low = drag_coefficient(0.1, coeffs, aspect_ratio=4.0)
        high = drag_coefficient(0.1, coeffs, aspect_ratio=20.0)
        assert low > high

    def test_is_stalled_flag(self):
        aircraft = FixedWing()
        state = np.zeros(12)
        state[2] = 100.0
        state[6] = 20.0
        state[8] = -20.0
        aircraft.reset(state=state)
        assert aircraft.is_stalled()


class TestPropulsion:
    def test_thrust_falls_off_with_airspeed(self):
        prop = PropulsionParams()
        fast, _ = propeller_thrust(0.6, 60.0, 1.2682, prop)
        slow, _ = propeller_thrust(0.6, 5.0, 1.2682, prop)
        assert slow > fast

    def test_more_throttle_gives_more_thrust(self):
        prop = PropulsionParams()
        low, _ = propeller_thrust(0.2, 30.0, 1.2682, prop)
        high, _ = propeller_thrust(0.9, 30.0, 1.2682, prop)
        assert high > low

    def test_airframe_wrench_excludes_thrust(self):
        wrench = airframe_wrench(
            velocity_body_frd=np.array([30.0, 0.0, 1.0]),
            rates_body_frd=np.zeros(3),
            surfaces=np.zeros(3),
            coeffs=AeroCoefficients(),
            wing_area=0.55,
            wing_span=2.9,
            chord=0.19,
            rho=1.2682,
        )
        assert wrench.thrust == 0.0


class TestTrim:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_every_preset_trims_at_cruise(self, preset):
        params = get_fixed_wing_params(preset)
        trim = compute_trim(params, airspeed=params.cruise_airspeed)
        assert trim.residual < 1e-3
        assert 0.0 <= trim.throttle <= 1.0

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_trimmed_flight_holds_altitude_open_loop(self, preset):
        """The strongest statement of correctness: trim is a real equilibrium."""
        aircraft = create_fixed_wing(preset)
        controls = aircraft.reset_trimmed(altitude=300.0)
        for _ in range(6000):
            aircraft.step(controls, 0.005)
        assert aircraft.state[2] == pytest.approx(300.0, abs=1.0)
        assert aircraft.airspeed == pytest.approx(aircraft.fw_params.cruise_airspeed, abs=0.5)

    def test_climb_trim_climbs(self):
        params = FixedWingParams()
        trim = compute_trim(params, airspeed=35.0, climb_rate=2.0)
        assert trim.residual < 1e-3
        # Climbing needs more throttle than level flight at the same speed.
        assert trim.throttle > compute_trim(params, airspeed=35.0).throttle

    def test_trim_below_stall_raises(self):
        params = FixedWingParams()
        with pytest.raises(TrimError):
            compute_trim(params, airspeed=0.4 * params.stall_airspeed)

    def test_trim_rejects_impossible_climb(self):
        with pytest.raises(TrimError):
            compute_trim(FixedWingParams(), airspeed=20.0, climb_rate=50.0)

    def test_higher_airspeed_needs_less_alpha(self):
        params = FixedWingParams()
        slow = compute_trim(params, airspeed=25.0)
        fast = compute_trim(params, airspeed=45.0)
        assert slow.alpha > fast.alpha


class TestPresets:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_stall_speed_is_below_cruise(self, preset):
        params = get_fixed_wing_params(preset)
        assert 0.0 < params.stall_airspeed < params.cruise_airspeed

    def test_overrides_apply(self):
        aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8, mass=5.0)
        assert aircraft.fw_params.mass == 5.0
        # The base preset must not be mutated by the override.
        assert get_fixed_wing_params(FixedWingPreset.SKYWALKER_X8).mass != 5.0

    def test_custom_preset_builds_from_kwargs(self):
        aircraft = create_fixed_wing(FixedWingPreset.CUSTOM, mass=2.0, wing_area=0.3)
        assert aircraft.fw_params.mass == 2.0

    def test_legacy_coefficient_aliases_still_read(self):
        params = FixedWingParams()
        assert params.CL0 == params.coeffs.CL0
        assert params.Cma == params.coeffs.Cma
        assert params.e_oswald == params.coeffs.e_oswald


class TestControlClamping:
    def test_surfaces_and_throttle_are_clamped(self):
        clamped = FixedWing.clamp_controls(np.array([10.0, -10.0, 10.0, 5.0]))
        assert abs(clamped[0]) <= np.radians(30.0) + 1e-9
        assert clamped[3] == 1.0

    def test_state_stays_finite_under_extreme_input(self):
        aircraft = FixedWing()
        aircraft.reset_trimmed(altitude=400.0)
        for _ in range(2000):
            aircraft.step(np.array([1.0, -1.0, 1.0, 1.0]), 0.005)
        assert np.all(np.isfinite(aircraft.state))


class TestAutopilot:
    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_holds_altitude_airspeed_and_course(self, preset):
        aircraft = create_fixed_wing(preset)
        aircraft.reset_trimmed(altitude=150.0)
        pilot = FixedWingAutopilot(aircraft.fw_params)
        cruise = aircraft.fw_params.cruise_airspeed
        command = AutopilotCommand(altitude=190.0, airspeed=cruise * 1.1, course=np.radians(50.0))
        dt = 0.01
        for _ in range(int(120 / dt)):
            aircraft.step(pilot.compute(aircraft.state, command, dt), dt)

        velocity = aircraft.velocity
        course = np.arctan2(velocity[1], velocity[0])
        course_error = np.degrees(
            np.arctan2(np.sin(course - command.course), np.cos(course - command.course))
        )
        assert aircraft.state[2] == pytest.approx(190.0, abs=3.0)
        assert aircraft.airspeed == pytest.approx(command.airspeed, abs=1.5)
        assert abs(course_error) < 3.0

    def test_gains_are_derived_not_hardcoded(self):
        """A heavier airframe must get different gains automatically."""
        light = FixedWingAutopilot(get_fixed_wing_params(FixedWingPreset.MINI_TRAINER))
        heavy = FixedWingAutopilot(get_fixed_wing_params(FixedWingPreset.CARGO_UAV))
        assert not np.isclose(light.kd_roll, heavy.kd_roll)

    def test_inner_loop_gains_have_stabilising_signs(self):
        """Rate feedback must never be positive feedback.

        Each derivative gain has to share the sign of its control
        authority, or be zero when the airframe is already damped past the
        design target.
        """
        for preset in ALL_PRESETS:
            pilot = FixedWingAutopilot(get_fixed_wing_params(preset))
            # Roll: aileron effectiveness is positive, so the gains are.
            assert pilot.kp_roll > 0.0
            assert pilot.kd_roll >= 0.0
            # Pitch: elevator effectiveness is negative, so the gains are.
            assert pilot.kp_pitch < 0.0
            assert pilot.kd_pitch <= 0.0

    def test_rejects_airframe_without_control_authority(self):
        coeffs = AeroCoefficients()
        coeffs.Clda = 0.0
        with pytest.raises(ValueError, match="roll control authority"):
            FixedWingAutopilot(FixedWingParams(coeffs=coeffs))

    def test_integrators_do_not_wind_up(self):
        """A long unreachable command must not leave a lasting overshoot."""
        aircraft = create_fixed_wing(FixedWingPreset.AEROSONDE)
        aircraft.reset_trimmed(altitude=150.0)
        pilot = FixedWingAutopilot(aircraft.fw_params)
        dt = 0.01

        unreachable = AutopilotCommand(altitude=4000.0, airspeed=35.0, course=0.0)
        for _ in range(int(60 / dt)):
            aircraft.step(pilot.compute(aircraft.state, unreachable, dt), dt)

        reachable = AutopilotCommand(altitude=float(aircraft.state[2]), airspeed=35.0, course=0.0)
        for _ in range(int(90 / dt)):
            aircraft.step(pilot.compute(aircraft.state, reachable, dt), dt)
        assert aircraft.state[2] == pytest.approx(reachable.altitude, abs=15.0)

    def test_integrator_is_held_while_saturated(self):
        """A command far outside the envelope must not charge the integrator."""
        params = get_fixed_wing_params(FixedWingPreset.AEROSONDE)
        pilot = FixedWingAutopilot(params)
        state = np.zeros(12)
        state[2] = 100.0
        state[6] = 35.0
        for _ in range(200):
            pilot.compute(state, AutopilotCommand(altitude=3000.0), 0.01)
        assert pilot._int_altitude == 0.0

    def test_reset_clears_integrators(self):
        params = get_fixed_wing_params(FixedWingPreset.AEROSONDE)
        pilot = FixedWingAutopilot(params)
        state = np.zeros(12)
        state[2] = 100.0
        state[6] = 35.0
        # A small, reachable error keeps the loop unsaturated so the
        # integrator actually accumulates.
        for _ in range(200):
            pilot.compute(state, AutopilotCommand(altitude=102.0), 0.01)
        assert pilot._int_altitude != 0.0
        pilot.reset()
        assert pilot._int_altitude == 0.0

    def test_custom_gains_are_honoured(self):
        params = get_fixed_wing_params(FixedWingPreset.AEROSONDE)
        tight = FixedWingAutopilot(params, AutopilotGains(max_roll_error=np.radians(5.0)))
        loose = FixedWingAutopilot(params, AutopilotGains(max_roll_error=np.radians(30.0)))
        assert tight.kp_roll > loose.kp_roll
