# Erwin Lejeune - 2026-08-18
"""Tests for fixed-wing mission waypoint navigation.

Almost every test here flies the real 6DOF airframe through the real
autopilot. The geometry tests at the top are the exception, and they exist
to pin the *signs*, which is where an ENU port of NED textbook equations
goes wrong — a guidance law with a flipped sign still converges, onto the
wrong side, in the wrong direction, and looks entirely plausible in a plot
until you check which way round the circle is being flown.

Tolerances are stated as a fraction of the airframe's own bank-limited turn
radius wherever the quantity scales with it, because 1 m of cross-track
error means something very different to a 0.6 kg trainer that turns in 15 m
and a 25 kg cargo aircraft that needs 104 m.
"""

from functools import lru_cache

import numpy as np
import pytest

from flybots.control.fixed_wing_autopilot import (
    AutopilotCommand,
    AutopilotGains,
    FixedWingAutopilot,
)
from flybots.guidance import (
    FixedWingMission,
    GuidanceError,
    GuidanceGains,
    LinePath,
    OrbitDirection,
    OrbitPath,
    minimum_turn_radius,
    orbit_plan,
    racetrack_plan,
    return_to_launch_plan,
    waypoint_plan,
    waypoint_reached,
)
from flybots.guidance.fixed_wing_mission import LineLeg, OrbitLeg
from flybots.vehicles.fixed_wing import (
    FixedWingPreset,
    create_fixed_wing,
    get_fixed_wing_params,
)

# Cruise speed each preset is flown at in these tests. The trainer and the
# X8 fly at their own cruise; the two big airframes are flown at cruise too,
# so every case is inside the trimmable envelope from the first step.
PRESET_SPEEDS = [
    (FixedWingPreset.MINI_TRAINER, 12.0),
    (FixedWingPreset.SKYWALKER_X8, 18.0),
    (FixedWingPreset.AEROSONDE, 35.0),
    (FixedWingPreset.CARGO_UAV, 32.0),
]

DT = 0.01


class Flight:
    """Recorded history of one closed-loop mission flight."""

    def __init__(self, positions, path_errors, labels, aircraft, mission):
        self.positions = np.asarray(positions)
        self.path_errors = np.asarray(path_errors)
        self.labels = labels
        self.aircraft = aircraft
        self.mission = mission
        self.times = np.arange(len(self.path_errors)) * DT

    def after(self, seconds: float) -> slice:
        """Index slice for everything past ``seconds``."""
        return slice(int(seconds / DT), None)


def fly(
    preset,
    legs,
    *,
    airspeed,
    start,
    heading=0.0,
    duration,
    loop=False,
    gains=None,
    dt=DT,
):
    """Fly ``legs`` with the real airframe, autopilot and mission manager."""
    aircraft = create_fixed_wing(preset)
    aircraft.reset_trimmed(airspeed=airspeed, altitude=float(start[2]), heading=heading)
    state = aircraft.state
    state[0], state[1] = float(start[0]), float(start[1])
    aircraft.reset(state=state)

    pilot = FixedWingAutopilot(aircraft.fw_params)
    mission = FixedWingMission(aircraft.fw_params, legs, gains, loop=loop)

    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    errors = np.zeros(steps)
    labels = []
    for i in range(steps):
        positions[i] = aircraft.state[:3]
        command = mission.update(aircraft.state)
        errors[i] = mission.diagnostics.path_error
        labels.append(mission.diagnostics.leg_label)
        aircraft.step(pilot.compute(aircraft.state, command, dt), dt)
        assert np.all(np.isfinite(aircraft.state)), f"diverged at t={i * dt:.1f}s"
    return Flight(positions, errors, labels, aircraft, mission)


def _wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


# ── geometry and signs, no simulation ────────────────────────────────────


class TestLineGeometry:
    def test_course_is_measured_counter_clockwise_from_east(self):
        """ENU, not NED: a line heading north has course +90 deg."""
        north = LinePath.between([0.0, 0.0, 100.0], [0.0, 500.0, 100.0], airspeed=20.0)
        assert np.degrees(north.course) == pytest.approx(90.0)

    def test_cross_track_is_positive_to_the_left(self):
        east = LinePath.between([0.0, 0.0, 100.0], [500.0, 0.0, 100.0], airspeed=20.0)
        assert east.cross_track([100.0, 40.0, 100.0]) == pytest.approx(40.0)
        assert east.cross_track([100.0, -40.0, 100.0]) == pytest.approx(-40.0)

    def test_left_of_the_path_steers_right(self):
        """The sign that decides whether the aircraft converges or diverges.

        In ENU, turning right *decreases* course. An aircraft to the left of
        an eastbound line must therefore be given a negative course.
        """
        east = LinePath.between([0.0, 0.0, 100.0], [500.0, 0.0, 100.0], airspeed=20.0)
        gains = GuidanceGains()
        assert east.command([0.0, 40.0, 100.0], gains).course < 0.0
        assert east.command([0.0, -40.0, 100.0], gains).course > 0.0

    def test_far_from_the_line_the_intercept_saturates(self):
        gains = GuidanceGains()
        east = LinePath.between([0.0, 0.0, 100.0], [500.0, 0.0, 100.0], airspeed=20.0)
        far = east.command([0.0, 1.0e6, 100.0], gains).course
        assert far == pytest.approx(-gains.course_infinity, abs=1e-3)

    def test_half_intercept_at_the_transition_distance(self):
        """``1 / k_path`` is defined as where half of chi_inf is commanded."""
        gains = GuidanceGains()
        east = LinePath.between([0.0, 0.0, 100.0], [500.0, 0.0, 100.0], airspeed=20.0)
        offset = gains.transition_distance(20.0)
        command = east.command([0.0, offset, 100.0], gains)
        assert command.course == pytest.approx(-gains.course_infinity / 2.0, rel=1e-9)

    def test_altitude_ramps_along_a_climbing_leg(self):
        climb = LinePath.between([0.0, 0.0, 100.0], [1000.0, 0.0, 200.0], airspeed=20.0)
        assert climb.altitude_at([0.0, 0.0, 0.0]) == pytest.approx(100.0)
        assert climb.altitude_at([500.0, 90.0, 0.0]) == pytest.approx(150.0)
        assert climb.altitude_at([1000.0, 0.0, 0.0]) == pytest.approx(200.0)

    def test_a_vertical_line_is_rejected(self):
        with pytest.raises(GuidanceError, match="horizontal component"):
            LinePath(origin=[0.0, 0.0, 0.0], direction=[0.0, 0.0, 1.0], airspeed=20.0)


class TestOrbitGeometry:
    @pytest.mark.parametrize(
        ("direction", "expected_deg"),
        [(OrbitDirection.COUNTER_CLOCKWISE, 90.0), (OrbitDirection.CLOCKWISE, -90.0)],
    )
    def test_on_the_circle_the_command_is_the_tangent(self, direction, expected_deg):
        """Due east of the centre, counter-clockwise means heading north."""
        orbit = OrbitPath([0.0, 0.0, 100.0], radius=300.0, direction=direction, airspeed=25.0)
        command = orbit.command([300.0, 0.0, 100.0], GuidanceGains())
        assert np.degrees(command.course) == pytest.approx(expected_deg)

    def test_far_outside_the_command_points_at_the_centre(self):
        orbit = OrbitPath(
            [0.0, 0.0, 100.0],
            radius=300.0,
            direction=OrbitDirection.COUNTER_CLOCKWISE,
            airspeed=25.0,
        )
        command = orbit.command([1.0e6, 0.0, 100.0], GuidanceGains())
        assert abs(np.degrees(command.course)) == pytest.approx(180.0, abs=0.1)

    def test_at_the_centre_the_command_points_straight_out(self):
        orbit = OrbitPath(
            [0.0, 0.0, 100.0],
            radius=300.0,
            direction=OrbitDirection.COUNTER_CLOCKWISE,
            airspeed=25.0,
        )
        # A hair east of the centre: radial error is -radius, which
        # saturates the law the other way.
        command = orbit.command([1e-6, 0.0, 100.0], GuidanceGains(loop_separation=0.02))
        assert abs(np.degrees(command.course)) < 5.0

    def test_radial_error_is_positive_outside(self):
        orbit = OrbitPath(
            [10.0, 20.0, 100.0],
            radius=300.0,
            direction=OrbitDirection.CLOCKWISE,
            airspeed=25.0,
        )
        assert orbit.radial_error([310.0, 20.0, 0.0]) == pytest.approx(0.0)
        assert orbit.radial_error([410.0, 20.0, 0.0]) == pytest.approx(100.0)

    def test_orbit_tighter_than_the_turn_radius_is_rejected(self):
        gains = GuidanceGains()
        minimum = gains.turn_radius(35.0)
        orbit = OrbitPath(
            [0.0, 0.0, 100.0],
            radius=0.5 * minimum,
            direction=OrbitDirection.CLOCKWISE,
            airspeed=35.0,
        )
        with pytest.raises(GuidanceError, match="bank-limited turn radius"):
            orbit.validate(gains)


class TestDerivedGains:
    def test_turn_radius_matches_the_coordinated_turn_relation(self):
        assert minimum_turn_radius(35.0, np.radians(45.0), 9.81) == pytest.approx(
            35.0**2 / 9.81, rel=1e-12
        )

    @pytest.mark.parametrize(("preset", "airspeed"), PRESET_SPEEDS)
    def test_transition_distance_clears_the_rollout_width(self, preset, airspeed):
        """The derived gain must not ask for a capture the bank limit forbids.

        No law can pull an aircraft onto a line from closer than the lateral
        distance it sweeps rolling out of a full intercept. If the derived
        transition distance ever dropped below that, the design would be
        internally inconsistent whatever the simulation showed.
        """
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        assert gains.transition_distance(airspeed) > gains.rollout_width(airspeed)

    def test_gains_follow_the_autopilot_rather_than_duplicating_it(self):
        """Tighten the autopilot's roll limit and the guidance retunes itself."""
        loose = GuidanceGains(autopilot=AutopilotGains(max_roll=np.radians(45.0)))
        tight = GuidanceGains(autopilot=AutopilotGains(max_roll=np.radians(20.0)))
        assert tight.turn_radius(35.0) > loose.turn_radius(35.0)
        # A shallower bank limit slows the course loop, so the guidance
        # outside it has to slow down too.
        assert tight.line_gain(35.0) < loose.line_gain(35.0)

    def test_slower_legs_get_a_tighter_gain(self):
        gains = GuidanceGains()
        assert gains.line_gain(12.0) > gains.line_gain(35.0)

    def test_orbit_and_line_converge_at_the_same_bandwidth(self):
        """One design number sets both laws; this is what that means."""
        gains = GuidanceGains()
        airspeed, radius = 25.0, 400.0
        line_pole = 2.0 * airspeed * gains.course_infinity * gains.line_gain(airspeed) / np.pi
        orbit_pole = airspeed * gains.orbit_gain(radius, airspeed) / radius
        assert line_pole == pytest.approx(orbit_pole, rel=1e-9)

    def test_a_bank_limit_outside_the_open_interval_is_rejected(self):
        with pytest.raises(GuidanceError, match="bank limit"):
            minimum_turn_radius(20.0, np.pi / 2)

    def test_course_infinity_at_ninety_degrees_is_rejected(self):
        with pytest.raises(GuidanceError, match="course_infinity"):
            GuidanceGains(course_infinity=np.pi / 2)


# ── waypoint acceptance ───────────────────────────────────────────────────


class TestWaypointAcceptance:
    """The radius / half-plane pair, tested as pure geometry."""

    NORTH_EAST = np.array([np.sqrt(0.5), np.sqrt(0.5)])

    def test_short_of_the_waypoint_and_outside_the_radius_is_not_reached(self):
        assert not waypoint_reached(
            [-200.0, 0.0, 100.0], [0.0, 0.0, 100.0], self.NORTH_EAST, capture_radius=50.0
        )

    def test_inside_the_capture_radius_is_reached(self):
        assert waypoint_reached(
            [-30.0, 0.0, 100.0], [0.0, 0.0, 100.0], self.NORTH_EAST, capture_radius=50.0
        )

    def test_an_overshoot_outside_the_radius_is_still_reached(self):
        """The whole reason the half-plane test exists.

        An aircraft pushed wide never enters the capture radius. With a
        radius test alone it turns back towards the waypoint, misses again
        on the other side, and orbits it forever. Crossing the half-plane
        settles the question: it is past the waypoint.
        """
        # Sailed 400 m past the corner and 100 m wide of the inbound leg:
        # eight capture radii away, and never any closer.
        overshot = [400.0, -100.0, 100.0]
        assert np.linalg.norm(overshot[:2]) > 50.0
        assert waypoint_reached(overshot, [0.0, 0.0, 100.0], self.NORTH_EAST, capture_radius=50.0)

    def test_altitude_is_not_a_gate(self):
        """A leg is finished when the aircraft is past it in plan view."""
        assert waypoint_reached(
            [10.0, 10.0, 40.0], [0.0, 0.0, 300.0], self.NORTH_EAST, capture_radius=1.0
        )

    def test_zero_radius_leaves_only_the_half_plane(self):
        assert not waypoint_reached(
            [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0], capture_radius=0.0
        )
        assert waypoint_reached([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0], capture_radius=0.0)


class TestPlanConstruction:
    def test_the_final_waypoint_of_an_open_mission_gets_no_fillet(self):
        gains = GuidanceGains()
        legs = waypoint_plan(
            [[0.0, 0.0, 100.0], [900.0, 0.0, 100.0], [900.0, 900.0, 100.0]],
            airspeed=25.0,
            gains=gains,
        )
        assert legs[-1].capture_radius == pytest.approx(0.0)

    def test_a_right_angle_corner_gets_one_turn_radius_of_fillet(self):
        """``R tan(delta/2)`` is ``R`` for a 90 degree turn."""
        gains = GuidanceGains()
        legs = waypoint_plan(
            [[0.0, 0.0, 100.0], [900.0, 0.0, 100.0], [900.0, 900.0, 100.0]],
            airspeed=25.0,
            gains=gains,
        )
        assert legs[0].capture_radius == pytest.approx(gains.turn_radius(25.0), rel=1e-9)

    def test_a_straight_through_waypoint_gets_no_fillet(self):
        gains = GuidanceGains()
        legs = waypoint_plan(
            [[0.0, 0.0, 100.0], [900.0, 0.0, 100.0], [1800.0, 0.0, 100.0]],
            airspeed=25.0,
            gains=gains,
        )
        assert legs[0].capture_radius == pytest.approx(0.0)

    def test_a_near_reversal_falls_back_to_the_incoming_direction(self):
        gains = GuidanceGains()
        legs = waypoint_plan(
            [[0.0, 0.0, 100.0], [900.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            airspeed=25.0,
            gains=gains,
        )
        np.testing.assert_allclose(legs[0].half_plane_normal, [1.0, 0.0], atol=1e-9)

    def test_looping_closes_the_circuit(self):
        gains = GuidanceGains()
        square = [[0.0, 0.0, 100.0], [900.0, 0.0, 100.0], [900.0, 900.0, 100.0]]
        assert len(waypoint_plan(square, airspeed=25.0, gains=gains)) == 2
        assert len(waypoint_plan(square, airspeed=25.0, gains=gains, loop=True)) == 3

    def test_a_single_waypoint_is_rejected(self):
        with pytest.raises(GuidanceError, match="at least two waypoints"):
            waypoint_plan([[0.0, 0.0, 100.0]], airspeed=25.0, gains=GuidanceGains())

    def test_repeated_waypoints_are_rejected(self):
        with pytest.raises(GuidanceError, match="same horizontal position"):
            waypoint_plan(
                [[0.0, 0.0, 100.0], [0.0, 0.0, 200.0]],
                airspeed=25.0,
                gains=GuidanceGains(),
            )

    def test_racetrack_straights_are_tangent_to_their_turns(self):
        """What makes the pattern close: each straight ends on the circle.

        If the straight did not arrive tangent, the hand-over to the
        half-orbit would step the commanded course and the aircraft would
        scallop through every turn.
        """
        gains = GuidanceGains()
        legs = racetrack_plan(
            [0.0, 0.0, 120.0],
            length=600.0,
            radius=200.0,
            heading=0.3,
            airspeed=25.0,
            gains=gains,
        )
        straight, far_turn, back, near_turn = legs
        # Each straight ends and the next begins on its turn's circle.
        assert far_turn.orbit.radial_error(straight.target) == pytest.approx(0.0, abs=1e-9)
        assert far_turn.orbit.radial_error(back.line.origin) == pytest.approx(0.0, abs=1e-9)
        assert near_turn.orbit.radial_error(back.target) == pytest.approx(0.0, abs=1e-9)
        # And the last turn closes the circuit onto the first straight.
        assert near_turn.orbit.radial_error(straight.line.origin) == pytest.approx(0.0, abs=1e-9)
        # Entry and exit of each half-orbit are diametrically opposite, which
        # is what makes the sweep exactly pi.
        entry = far_turn.orbit.angle_at(straight.target)
        exit_ = far_turn.orbit.angle_at(back.line.origin)
        assert abs(_wrap(exit_ - entry)) == pytest.approx(np.pi, abs=1e-9)

    def test_racetrack_straights_are_two_radii_apart(self):
        legs = racetrack_plan(
            [0.0, 0.0, 120.0],
            length=600.0,
            radius=200.0,
            heading=0.0,
            airspeed=25.0,
            gains=GuidanceGains(),
        )
        outbound, _, inbound, _ = legs
        assert outbound.line.origin[1] == pytest.approx(-200.0)
        assert inbound.line.origin[1] == pytest.approx(200.0)

    @pytest.mark.parametrize(
        ("direction", "expected_sign"),
        [(OrbitDirection.COUNTER_CLOCKWISE, -1.0), (OrbitDirection.CLOCKWISE, +1.0)],
    )
    def test_orbits_feed_forward_the_coordinated_turn_bank(self, direction, expected_sign):
        """Left turns want left bank, which is negative roll in ENU."""
        gains = GuidanceGains()
        orbit = OrbitPath([0.0, 0.0, 120.0], radius=400.0, direction=direction, airspeed=25.0)
        output = orbit.command([400.0, 0.0, 120.0], gains)
        expected = expected_sign * np.arctan(25.0**2 / (gains.gravity * 400.0))
        assert output.roll_feedforward == pytest.approx(expected)
        # And that bank really does produce the orbit's turn rate.
        turn_rate = -gains.gravity * np.tan(output.roll_feedforward) / 25.0
        assert turn_rate == pytest.approx(int(direction) * 25.0 / 400.0, rel=1e-9)

    def test_straight_legs_feed_forward_nothing(self):
        line = LinePath.between([0.0, 0.0, 120.0], [900.0, 0.0, 120.0], airspeed=25.0)
        assert line.command([0.0, 40.0, 120.0], GuidanceGains()).roll_feedforward == 0.0

    def test_racetrack_half_orbits_sweep_half_a_turn(self):
        legs = racetrack_plan(
            [0.0, 0.0, 120.0],
            length=600.0,
            radius=200.0,
            heading=0.0,
            airspeed=25.0,
            gains=GuidanceGains(),
        )
        assert legs[1].sweep == pytest.approx(np.pi)
        assert legs[3].sweep == pytest.approx(np.pi)

    def test_racetrack_rejects_a_turn_the_airframe_cannot_fly(self):
        gains = GuidanceGains()
        with pytest.raises(GuidanceError, match="bank-limited turn radius"):
            racetrack_plan(
                [0.0, 0.0, 120.0],
                length=600.0,
                radius=0.5 * gains.turn_radius(35.0),
                heading=0.0,
                airspeed=35.0,
                gains=gains,
            )

    def test_rtl_from_inside_the_hold_skips_the_transit(self):
        gains = GuidanceGains()
        legs = return_to_launch_plan(
            [10.0, 10.0, 90.0],
            [0.0, 0.0, 100.0],
            safe_altitude=150.0,
            airspeed=25.0,
            gains=gains,
        )
        assert [leg.label for leg in legs] == ["loiter"]

    def test_rtl_transits_at_the_safe_altitude(self):
        legs = return_to_launch_plan(
            [4000.0, 0.0, 40.0],
            [0.0, 0.0, 100.0],
            safe_altitude=250.0,
            airspeed=25.0,
            gains=GuidanceGains(),
        )
        transit, loiter = legs
        # Level at the safe altitude from the first instant, so the climb
        # starts immediately instead of being scheduled as a phase.
        assert transit.line.climb_gradient == pytest.approx(0.0)
        assert transit.line.altitude_at([4000.0, 0.0, 40.0]) == pytest.approx(250.0)
        assert loiter.orbit.altitude == pytest.approx(250.0)

    def test_rtl_hands_over_at_the_holding_circle(self):
        gains = GuidanceGains()
        legs = return_to_launch_plan(
            [4000.0, 0.0, 40.0],
            [0.0, 0.0, 100.0],
            safe_altitude=250.0,
            airspeed=25.0,
            gains=gains,
        )
        assert legs[0].capture_radius == pytest.approx(legs[1].orbit.radius)
        assert legs[1].orbit.radius == pytest.approx(2.0 * gains.turn_radius(25.0))


# ── closed-loop flight ────────────────────────────────────────────────────


class TestStraightLineFollowing:
    @pytest.mark.parametrize(("preset", "airspeed"), PRESET_SPEEDS)
    def test_converges_onto_the_line_and_stays_there(self, preset, airspeed):
        """Start five turn radii off the line and end on it.

        The assertion that matters is the *sustained* error at the end, not
        that the aircraft crossed the line at some point. A law with the
        cross-track sign wrong diverges; a law that points the nose at the
        far waypoint instead converges to an offset and sits there.
        """
        params = get_fixed_wing_params(preset)
        gains = GuidanceGains(gravity=params.gravity)
        radius = gains.turn_radius(airspeed)
        # Long enough for the capture transient to have run out on the
        # slowest airframe: the linearised cross-track time constant is
        # 1 / convergence_bandwidth, and this is roughly ten of them.
        duration = 55.0 * radius / airspeed

        flight = fly(
            preset,
            waypoint_plan(
                [[0.0, 0.0, 150.0], [400.0 * radius, 0.0, 150.0]],
                airspeed=airspeed,
                gains=gains,
            ),
            airspeed=airspeed,
            start=(0.0, 5.0 * radius, 150.0),
            duration=duration,
        )

        settled = flight.path_errors[flight.after(0.85 * duration)]
        assert flight.path_errors[0] == pytest.approx(5.0 * radius, rel=0.02)
        # 2 % of the turn radius, against a measured 0.43-0.59 % on these
        # four airframes — a tolerance with margin, not one fitted to the
        # answer.
        assert np.abs(settled).max() < 0.02 * radius
        # Approaching from the left, an overshoot shows up as a negative
        # excursion. The derived gain is sized so there is not one: the
        # measured minimum is positive on every preset.
        assert flight.path_errors.min() > -0.05 * radius

    def test_a_climbing_leg_arrives_at_the_target_altitude(self):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        flight = fly(
            preset,
            waypoint_plan(
                [[0.0, 0.0, 80.0], [900.0, 0.0, 140.0]],
                airspeed=airspeed,
                gains=gains,
            ),
            airspeed=airspeed,
            start=(0.0, 0.0, 80.0),
            duration=75.0,
        )
        # 900 m at 12 m/s is 75 s, so the aircraft is at the far waypoint.
        assert flight.positions[-1, 2] == pytest.approx(140.0, abs=3.0)
        assert flight.positions[-1, 0] > 800.0

    def test_pointing_at_the_waypoint_would_not_pass_this(self):
        """A displaced start is corrected, not merely aimed away from.

        Flown from a position lateral to the line but with the far waypoint
        almost dead ahead, a chase-the-point law is already satisfied and
        leaves the offset standing. The cross-track law removes it.
        """
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        offset = 4.0 * gains.turn_radius(airspeed)
        flight = fly(
            preset,
            waypoint_plan(
                [[0.0, 0.0, 120.0], [6000.0, 0.0, 120.0]],
                airspeed=airspeed,
                gains=gains,
            ),
            airspeed=airspeed,
            start=(0.0, offset, 120.0),
            duration=90.0,
        )
        assert abs(flight.path_errors[-1]) < 0.05 * offset


class TestOrbit:
    @pytest.mark.parametrize(
        "direction", [OrbitDirection.COUNTER_CLOCKWISE, OrbitDirection.CLOCKWISE]
    )
    def test_holds_the_commanded_radius(self, direction):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        radius = 3.0 * gains.turn_radius(airspeed)
        centre = np.array([0.0, 0.0, 120.0])

        flight = fly(
            preset,
            orbit_plan(centre, radius=radius, airspeed=airspeed, gains=gains, direction=direction),
            airspeed=airspeed,
            # Start well outside and pointing away, so the orbit has to be
            # captured rather than merely maintained.
            start=(3.0 * radius, 0.0, 120.0),
            heading=0.0,
            duration=180.0,
        )

        settled = flight.path_errors[flight.after(120.0)]
        assert np.abs(settled).max() < 0.03 * radius
        assert flight.positions[flight.after(120.0), 2] == pytest.approx(120.0, abs=3.0)

    @pytest.mark.parametrize(
        ("direction", "sign"),
        [(OrbitDirection.COUNTER_CLOCKWISE, +1.0), (OrbitDirection.CLOCKWISE, -1.0)],
    )
    def test_flies_the_commanded_direction(self, direction, sign):
        """ENU: counter-clockwise means the polar angle increases.

        This is the test that a NED sign left in place would fail — the
        aircraft would hold the radius perfectly, going the wrong way round.
        """
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        radius = 3.0 * gains.turn_radius(airspeed)
        centre = np.array([0.0, 0.0, 120.0])

        flight = fly(
            preset,
            orbit_plan(centre, radius=radius, airspeed=airspeed, gains=gains, direction=direction),
            airspeed=airspeed,
            start=(radius, 0.0, 120.0),
            heading=sign * np.pi / 2.0,
            duration=120.0,
        )
        offsets = flight.positions[:, :2] - centre[:2]
        angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
        assert sign * (angles[-1] - angles[0]) > 2.0 * np.pi

    def test_captures_the_orbit_from_the_centre(self):
        """Started at the centre, the field commands straight out and spirals on."""
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        radius = 3.0 * gains.turn_radius(airspeed)
        flight = fly(
            preset,
            orbit_plan([0.0, 0.0, 120.0], radius=radius, airspeed=airspeed, gains=gains),
            airspeed=airspeed,
            start=(0.0, 0.0, 120.0),
            duration=180.0,
        )
        assert np.abs(flight.path_errors[flight.after(120.0)]).max() < 0.03 * radius


@lru_cache(maxsize=1)
def _racetrack_circuit():
    """A four-lap mini-trainer racetrack of 2.5 turn radii, flown once."""
    preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
    gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
    radius = 2.5 * gains.turn_radius(airspeed)
    length = 8.0 * gains.turn_radius(airspeed)
    legs = racetrack_plan(
        [0.0, 0.0, 120.0],
        length=length,
        radius=radius,
        heading=0.0,
        airspeed=airspeed,
        gains=gains,
    )
    start = legs[0].line.origin
    flight = fly(
        preset,
        legs,
        airspeed=airspeed,
        start=(start[0], start[1], 120.0),
        heading=0.0,
        duration=320.0,
        loop=True,
    )
    return flight, radius, length


class TestRacetrack:
    """A racetrack of 2.5 turn radii — tight enough to be a real test."""

    PRESET, AIRSPEED = FixedWingPreset.MINI_TRAINER, 12.0

    @staticmethod
    def circuit():
        """Four laps of the same circuit, flown once and shared.

        Every assertion in this class is about the same flight seen from a
        different angle, so flying it once rather than four times keeps the
        suite honest about what it costs.
        """
        return _racetrack_circuit()

    @staticmethod
    def lap_starts(flight):
        """Indices where the circuit rolls over onto its first leg."""
        labels = np.array(flight.labels)
        changed = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        return [i for i in changed if labels[i] == "outbound"]

    def test_the_pattern_closes(self):
        """Successive laps start from the same point, not a drifting one.

        Timing is deliberately not used to find "the same point in the
        circuit": the aircraft flies a slightly longer path than the ideal
        perimeter, so a fixed lap time would compare two different places
        and call the difference drift. The leg the mission manager reports
        is the ground truth for where in the pattern it is.
        """
        flight, radius, _ = self.circuit()
        starts = self.lap_starts(flight)
        assert len(starts) >= 3, "fewer than three laps flown"

        drifts = [
            float(np.linalg.norm(flight.positions[b, :2] - flight.positions[a, :2]))
            for a, b in zip(starts[1:-1], starts[2:], strict=True)
        ]
        assert max(drifts) < 0.05 * radius
        assert np.abs(flight.positions[:, 2] - 120.0).max() < 5.0

    def test_the_pattern_stays_inside_its_own_envelope(self):
        """No bulge outside the geometry, beyond a stated turn-entry margin.

        The margin is not zero and cannot be. Entering a turn from a
        straight steps the required turn rate, and even with the bank fed
        forward the roll takes finite time, so the aircraft always sits a
        little outside the circle for the first part of each turn. The
        assertion is that the bulge stays *small* — a quarter of the turn
        radius — not that it is absent.
        """
        flight, radius, length = self.circuit()
        # Skip the first lap, where the aircraft is still settling on.
        tail = flight.positions[flight.after(60.0)]
        assert np.abs(tail[:, 0]).max() < 0.5 * length + 1.25 * radius
        assert np.abs(tail[:, 1]).max() < 1.25 * radius

    def test_every_leg_of_the_circuit_is_flown(self):
        flight, _, _ = self.circuit()
        assert set(flight.labels) == {"outbound", "far turn", "inbound", "near turn"}

    def test_the_turns_go_the_way_they_were_asked_to(self):
        """A clockwise circuit must run clockwise, ENU signs and all."""
        gains = GuidanceGains(gravity=get_fixed_wing_params(self.PRESET).gravity)
        radius = 2.5 * gains.turn_radius(self.AIRSPEED)
        length = 8.0 * gains.turn_radius(self.AIRSPEED)
        for direction, sign in (
            (OrbitDirection.COUNTER_CLOCKWISE, +1.0),
            (OrbitDirection.CLOCKWISE, -1.0),
        ):
            legs = racetrack_plan(
                [0.0, 0.0, 120.0],
                length=length,
                radius=radius,
                heading=0.0,
                airspeed=self.AIRSPEED,
                gains=gains,
                direction=direction,
            )
            start = legs[0].line.origin
            flight = fly(
                self.PRESET,
                legs,
                airspeed=self.AIRSPEED,
                start=(start[0], start[1], 120.0),
                duration=80.0,
                loop=True,
            )
            offsets = flight.positions[:, :2]
            angles = np.unwrap(np.arctan2(offsets[:, 1], offsets[:, 0]))
            assert sign * (angles[-1] - angles[0]) > np.pi


class TestTurnFeedForward:
    """The one additive change to the autopilot, tested on its own.

    The guidance layer hands the autopilot the bank a curved path needs.
    These pin the two properties that make that safe: it is inert when the
    guidance does not use it, and it can never spend more bank than the
    envelope allows.
    """

    @staticmethod
    def level_state(course=0.0, airspeed=12.0):
        state = np.zeros(12)
        state[2] = 120.0
        state[5] = course
        state[6] = airspeed
        return state

    def test_the_default_leaves_the_loop_a_pure_feedback_design(self):
        """Zero feed-forward must reproduce the course PI exactly."""
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        pilot = FixedWingAutopilot(params)
        state = self.level_state()
        pilot.compute(state, AutopilotCommand(altitude=120.0, airspeed=12.0, course=0.3), DT)
        # First step: the integrator has barely charged, so the command is
        # the proportional term, negated for ENU.
        assert pilot.diagnostics.roll_cmd == pytest.approx(-pilot.kp_course * 0.3, abs=1e-3)

    def test_a_fed_forward_bank_is_commanded_when_the_course_is_already_right(self):
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        pilot = FixedWingAutopilot(params)
        command = AutopilotCommand(
            altitude=120.0, airspeed=12.0, course=0.0, roll_feedforward=-0.3
        )
        pilot.compute(self.level_state(), command, DT)
        assert pilot.diagnostics.roll_cmd == pytest.approx(-0.3)

    def test_the_total_roll_command_stays_inside_the_envelope(self):
        """The PI is given only the authority the feed-forward has not spent."""
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        gains = AutopilotGains()
        pilot = FixedWingAutopilot(params, gains)
        for feedforward in (-gains.max_roll, -0.3, 0.0, 0.3, gains.max_roll):
            pilot.reset()
            command = AutopilotCommand(
                altitude=120.0,
                airspeed=12.0,
                course=np.radians(150.0),
                roll_feedforward=feedforward,
            )
            for _ in range(400):
                pilot.compute(self.level_state(), command, DT)
            assert abs(pilot.diagnostics.roll_cmd) <= gains.max_roll + 1e-9

    def test_an_absurd_feedforward_is_clamped_not_obeyed(self):
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        gains = AutopilotGains()
        pilot = FixedWingAutopilot(params, gains)
        command = AutopilotCommand(
            altitude=120.0, airspeed=12.0, course=0.0, roll_feedforward=10.0
        )
        pilot.compute(self.level_state(), command, DT)
        assert pilot.diagnostics.roll_cmd == pytest.approx(gains.max_roll)


class TestReturnToLaunch:
    def test_climbs_flies_home_and_loiters(self):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        params = get_fixed_wing_params(preset)
        gains = GuidanceGains(gravity=params.gravity)
        home = np.array([0.0, 0.0, 60.0])
        safe_altitude = 130.0
        loiter_radius = 2.0 * gains.turn_radius(airspeed)

        flight = fly(
            preset,
            return_to_launch_plan(
                [700.0, 500.0, 60.0],
                home,
                safe_altitude=safe_altitude,
                airspeed=airspeed,
                gains=gains,
            ),
            airspeed=airspeed,
            start=(700.0, 500.0, 60.0),
            # Pointing away from home, so it has to turn round first.
            heading=np.radians(40.0),
            duration=200.0,
        )

        tail = flight.positions[flight.after(140.0)]
        distance = np.linalg.norm(tail[:, :2] - home[:2], axis=1)
        assert distance.max() < 1.3 * loiter_radius
        assert tail[:, 2] == pytest.approx(safe_altitude, abs=5.0)
        assert flight.mission.diagnostics.leg_label == "loiter"

    def test_the_climb_starts_immediately_rather_than_at_home(self):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        flight = fly(
            preset,
            return_to_launch_plan(
                [700.0, 0.0, 60.0],
                [0.0, 0.0, 60.0],
                safe_altitude=130.0,
                airspeed=airspeed,
                gains=gains,
            ),
            airspeed=airspeed,
            start=(700.0, 0.0, 60.0),
            heading=np.pi,
            duration=90.0,
        )
        # Well before arriving, the aircraft is already at height.
        assert flight.positions[int(60.0 / DT), 2] > 120.0

    def test_triggering_rtl_mid_mission_abandons_the_plan(self):
        """RTL has to work from wherever the aircraft happens to be."""
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        params = get_fixed_wing_params(preset)
        gains = GuidanceGains(gravity=params.gravity)
        home = np.array([0.0, 0.0, 100.0])

        aircraft = create_fixed_wing(preset)
        aircraft.reset_trimmed(airspeed=airspeed, altitude=100.0, heading=0.0)
        pilot = FixedWingAutopilot(aircraft.fw_params)
        mission = FixedWingMission(
            aircraft.fw_params,
            waypoint_plan(
                [home, [900.0, 0.0, 100.0], [900.0, 900.0, 100.0]],
                airspeed=airspeed,
                gains=gains,
            ),
            gains,
            home=home,
        )

        for step in range(int(220.0 / DT)):
            if step == int(60.0 / DT):
                mission.return_to_launch(aircraft.state, safe_altitude=160.0, airspeed=airspeed)
                assert mission.diagnostics.leg_label == "transit home"
            aircraft.step(pilot.compute(aircraft.state, mission.update(aircraft.state), DT), DT)

        assert mission.diagnostics.leg_label == "loiter"
        assert np.linalg.norm(aircraft.state[:2] - home[:2]) < 3.0 * gains.turn_radius(airspeed)
        assert aircraft.state[2] == pytest.approx(160.0, abs=6.0)

    def test_rtl_without_a_home_is_refused(self):
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        mission = FixedWingMission(
            params,
            orbit_plan([0.0, 0.0, 100.0], radius=200.0, airspeed=12.0, gains=GuidanceGains()),
        )
        with pytest.raises(GuidanceError, match="needs a home position"):
            mission.return_to_launch(np.zeros(12), safe_altitude=150.0)


class TestMissionSequencing:
    def test_every_waypoint_is_visited_in_order(self):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        waypoints = [
            [0.0, 0.0, 120.0],
            [400.0, 0.0, 120.0],
            [400.0, 400.0, 120.0],
            [0.0, 400.0, 120.0],
        ]
        flight = fly(
            preset,
            waypoint_plan(waypoints, airspeed=airspeed, gains=gains),
            airspeed=airspeed,
            start=(0.0, 0.0, 120.0),
            duration=160.0,
        )
        assert flight.labels[0] == "leg 1"
        assert [label for label in dict.fromkeys(flight.labels)] == ["leg 1", "leg 2", "leg 3"]
        assert flight.mission.is_complete
        # Every waypoint was passed within a fillet radius of itself.
        for waypoint in waypoints[1:]:
            closest = np.linalg.norm(
                flight.positions[:, :2] - np.array(waypoint[:2]), axis=1
            ).min()
            assert closest < 1.5 * gains.turn_radius(airspeed)

    def test_an_overshot_corner_does_not_trap_the_mission(self):
        """The behaviour the half-plane test buys, flown rather than asserted.

        These waypoints ask for a 150 degree reversal in a space far tighter
        than the airframe's turn radius, so the aircraft cannot possibly
        make the corner and sails past. A radius-only acceptance test would
        leave it circling the missed waypoint for the rest of the flight.
        """
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        radius = gains.turn_radius(airspeed)
        waypoints = [
            [0.0, 0.0, 120.0],
            [30.0 * radius, 0.0, 120.0],
            [
                30.0 * radius - 40.0 * radius * np.cos(np.radians(30.0)),
                40.0 * radius * np.sin(np.radians(30.0)),
                120.0,
            ],
        ]
        flight = fly(
            preset,
            waypoint_plan(waypoints, airspeed=airspeed, gains=gains),
            airspeed=airspeed,
            start=(0.0, 0.0, 120.0),
            duration=260.0,
        )
        assert "leg 2" in flight.labels, "mission never left the first leg"
        assert flight.mission.is_complete

    def test_a_looping_mission_repeats_indefinitely(self):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        flight = fly(
            preset,
            waypoint_plan(
                [
                    [0.0, 0.0, 120.0],
                    [400.0, 0.0, 120.0],
                    [400.0, 400.0, 120.0],
                    [0.0, 400.0, 120.0],
                ],
                airspeed=airspeed,
                gains=gains,
                loop=True,
            ),
            airspeed=airspeed,
            start=(0.0, 0.0, 120.0),
            duration=200.0,
            loop=True,
        )
        assert flight.mission.diagnostics.laps >= 1
        assert not flight.mission.is_complete

    def test_a_finished_mission_keeps_flying_the_last_leg(self):
        preset, airspeed = FixedWingPreset.MINI_TRAINER, 12.0
        gains = GuidanceGains(gravity=get_fixed_wing_params(preset).gravity)
        flight = fly(
            preset,
            waypoint_plan(
                [[0.0, 0.0, 120.0], [300.0, 0.0, 120.0]], airspeed=airspeed, gains=gains
            ),
            airspeed=airspeed,
            start=(0.0, 0.0, 120.0),
            duration=60.0,
        )
        assert flight.mission.is_complete
        assert flight.positions[-1, 0] > 300.0
        assert abs(flight.path_errors[-1]) < 5.0

    def test_diagnostics_track_the_active_leg(self):
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        gains = GuidanceGains(gravity=params.gravity)
        mission = FixedWingMission(
            params,
            waypoint_plan([[0.0, 0.0, 120.0], [900.0, 0.0, 120.0]], airspeed=12.0, gains=gains),
        )
        state = np.zeros(12)
        state[:3] = [10.0, 25.0, 120.0]
        command = mission.update(state)
        assert mission.diagnostics.leg_index == 0
        assert mission.diagnostics.path_error == pytest.approx(25.0)
        assert command.course == pytest.approx(mission.diagnostics.course_command)
        assert command.altitude == pytest.approx(120.0)

    def test_an_empty_mission_is_refused(self):
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        mission = FixedWingMission(params)
        with pytest.raises(GuidanceError, match="no legs"):
            mission.update(np.zeros(12))
        with pytest.raises(GuidanceError, match="at least one leg"):
            mission.fly([])

    def test_reset_rewinds_to_the_first_leg(self):
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        gains = GuidanceGains(gravity=params.gravity)
        mission = FixedWingMission(
            params,
            waypoint_plan(
                [[0.0, 0.0, 120.0], [900.0, 0.0, 120.0], [900.0, 900.0, 120.0]],
                airspeed=12.0,
                gains=gains,
            ),
        )
        state = np.zeros(12)
        state[:3] = [1200.0, 100.0, 120.0]
        mission.update(state)
        assert mission.diagnostics.leg_index == 1
        mission.reset()
        assert mission.diagnostics.leg_index == 0

    def test_orbit_legs_terminate_on_swept_angle(self):
        """A half-orbit ends after half a turn, not on a position test."""
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        gains = GuidanceGains(gravity=params.gravity)
        orbit = OrbitPath(
            [0.0, 0.0, 120.0],
            radius=100.0,
            direction=OrbitDirection.COUNTER_CLOCKWISE,
            airspeed=12.0,
        )
        after = LineLeg(
            line=LinePath(origin=[0.0, 100.0, 120.0], direction=[-1.0, 0.0, 0.0], airspeed=12.0),
            target=[-900.0, 100.0, 120.0],
            half_plane_normal=[-1.0, 0.0],
            capture_radius=0.0,
            label="after",
        )
        mission = FixedWingMission(params, [OrbitLeg(orbit=orbit, sweep=np.pi), after], gains)

        state = np.zeros(12)
        for angle in np.linspace(0.0, np.pi, 60):
            state[:3] = [100.0 * np.cos(angle), 100.0 * np.sin(angle), 120.0]
            mission.update(state)
        assert mission.diagnostics.leg_label == "after"

    def test_swept_angle_does_not_jump_across_the_branch_cut(self):
        """Crossing +-pi must not be mistaken for most of a lap."""
        params = get_fixed_wing_params(FixedWingPreset.MINI_TRAINER)
        gains = GuidanceGains(gravity=params.gravity)
        orbit = OrbitPath(
            [0.0, 0.0, 120.0],
            radius=100.0,
            direction=OrbitDirection.COUNTER_CLOCKWISE,
            airspeed=12.0,
        )
        mission = FixedWingMission(params, [OrbitLeg(orbit=orbit, sweep=4.0 * np.pi)], gains)
        state = np.zeros(12)
        for angle in np.linspace(np.pi - 0.4, np.pi + 0.4, 40):
            state[:3] = [100.0 * np.cos(angle), 100.0 * np.sin(angle), 120.0]
            mission.update(state)
        assert mission.diagnostics.angle_flown == pytest.approx(0.8, abs=0.05)
