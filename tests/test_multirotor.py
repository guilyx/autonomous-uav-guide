# Erwin Lejeune - 2026-08-18
"""Tests for the general N-rotor multirotor model.

These assert on flight, not on shape. A hexacopter that hovers with zero
drift, rolls the right way and yaws out of its spin pattern is a
hexacopter; one whose ``step()`` returns twelve floats is not.
"""

import numpy as np
import pytest

from flybots.vehicles.components.allocation import Rotor, coaxial_layout, x_layout
from flybots.vehicles.multirotor import (
    Multirotor,
    MultirotorParams,
    Quadrotor,
    QuadrotorParams,
)
from flybots.vehicles.presets import (
    VehiclePreset,
    create_multirotor,
    create_quadrotor,
    get_params,
)

DT = 0.001

MULTIROTOR_PRESETS = [
    VehiclePreset.RACING_250,
    VehiclePreset.HEX_S550,
    VehiclePreset.OCTO_X8,
]


def _hovering(preset: VehiclePreset, altitude: float = 5.0) -> Multirotor:
    """An aircraft at *altitude* with its motors already at hover speed."""
    aircraft = create_multirotor(preset)
    aircraft.reset(position=np.array([0.0, 0.0, altitude]))
    aircraft.spin_up_to_hover()
    return aircraft


class TestQuadrotorIsUnchanged:
    """The four-rotor case has to fall out of the general model untouched."""

    def test_quadrotor_is_a_multirotor(self):
        assert issubclass(Quadrotor, Multirotor)
        assert isinstance(Quadrotor(), Multirotor)

    def test_default_construction_still_takes_quadrotor_params(self):
        quad = Quadrotor(QuadrotorParams(mass=2.0, arm_length=0.2))
        assert quad.params.mass == 2.0
        assert quad.params.arm_length == 0.2
        assert quad.params.frame == "x"

    def test_a_quadrotor_has_four_rotors_and_four_motors(self):
        quad = Quadrotor()
        assert quad.n_rotors == 4
        assert len(quad.motors) == 4
        assert len(quad.rotors) == 4

    def test_the_mixer_still_exposes_the_frame(self):
        assert Quadrotor(QuadrotorParams(frame="+")).mixer.frame == "+"

    def test_motor_spin_directions_follow_the_geometry(self):
        """Reaction torque signs come from the layout, not a separate list."""
        quad = Quadrotor()
        assert [m.direction for m in quad.motors] == [r.direction for r in quad.rotors]
        assert [m.direction for m in quad.motors] == [1, -1, 1, -1]

    def test_hover_holds_altitude(self):
        quad = _hovering(VehiclePreset.RACING_250, altitude=1.0)
        for _ in range(1000):
            quad.step(quad.hover_wrench(), DT)
        assert abs(quad.position[2] - 1.0) < 1e-3


class TestHover:
    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_hover_holds_altitude_with_no_drift(self, preset: VehiclePreset):
        """Five seconds of open-loop hover: nothing should move at all.

        Any residual torque in the layout — a spin pattern that does not
        cancel, a rotor placed off its arm — shows up here as yaw creep or
        a slow lean long before it shows up in a controller.
        """
        aircraft = _hovering(preset)
        for _ in range(5000):
            aircraft.step(aircraft.hover_wrench(), DT)

        assert aircraft.position[2] == pytest.approx(5.0, abs=1e-3)
        np.testing.assert_allclose(aircraft.position[:2], 0.0, atol=1e-6)
        np.testing.assert_allclose(aircraft.euler, 0.0, atol=1e-9)
        np.testing.assert_allclose(aircraft.angular_velocity, 0.0, atol=1e-9)

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_hover_thrust_matches_weight(self, preset: VehiclePreset):
        aircraft = _hovering(preset)
        params = get_params(preset)
        assert aircraft.get_rotor_thrusts().sum() == pytest.approx(
            params.mass * params.gravity, rel=1e-9
        )

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_every_rotor_is_well_inside_its_limits_at_hover(self, preset: VehiclePreset):
        aircraft = _hovering(preset)
        params = get_params(preset)
        assert np.all(aircraft.get_motor_speeds() < 0.75 * params.omega_max)
        assert params.thrust_to_weight > 2.0

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_freefall_accelerates_downward_at_g(self, preset: VehiclePreset):
        aircraft = create_multirotor(preset)
        aircraft.reset(position=np.array([0.0, 0.0, 50.0]))
        for _ in range(100):
            aircraft.step(np.zeros(4), DT)
        assert aircraft.velocity[2] == pytest.approx(-9.81 * 0.1, rel=0.02)


class TestAttitudeResponseSigns:
    """FLU, and only FLU. Every one of these flips in a textbook FRD frame."""

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_positive_roll_torque_banks_right(self, preset: VehiclePreset):
        aircraft = _hovering(preset)
        wrench = aircraft.hover_wrench()
        arm = aircraft.rotors[0].arm_length
        wrench[1] = 0.05 * wrench[0] * arm
        for _ in range(200):
            aircraft.step(wrench, DT)
        assert aircraft.euler[0] > 0.05
        assert aircraft.angular_velocity[0] > 0.0

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_a_right_bank_is_flown_by_the_left_hand_rotors(self, preset: VehiclePreset):
        """tau_x = sum(y_i f_i): rolling right means lifting the left side."""
        aircraft = create_multirotor(preset)
        wrench = aircraft.hover_wrench()
        arm = aircraft.rotors[0].arm_length
        forces = aircraft.mixer.wrench_to_forces(
            np.array([wrench[0], 0.05 * wrench[0] * arm, 0.0, 0.0])
        )
        left = np.array([f for f, r in zip(forces, aircraft.rotors) if r.y > 1e-9])
        right = np.array([f for f, r in zip(forces, aircraft.rotors) if r.y < -1e-9])
        assert left.mean() > right.mean()

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_positive_pitch_torque_pitches_nose_down(self, preset: VehiclePreset):
        """Positive theta is nose-down in FLU, and it is flown from the rear."""
        aircraft = _hovering(preset)
        wrench = aircraft.hover_wrench()
        arm = aircraft.rotors[0].arm_length
        wrench[2] = 0.05 * wrench[0] * arm
        for _ in range(200):
            aircraft.step(wrench, DT)
        assert aircraft.euler[1] > 0.05

        forces = aircraft.mixer.wrench_to_forces(wrench)
        rear = np.array([f for f, r in zip(forces, aircraft.rotors) if r.x < -1e-9])
        front = np.array([f for f, r in zip(forces, aircraft.rotors) if r.x > 1e-9])
        assert rear.mean() > front.mean()

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_positive_yaw_torque_turns_counter_clockwise(self, preset: VehiclePreset):
        aircraft = _hovering(preset)
        wrench = aircraft.hover_wrench()
        wrench[3] = 0.02 * wrench[0] * aircraft.rotors[0].arm_length
        for _ in range(300):
            aircraft.step(wrench, DT)
        assert aircraft.euler[2] > 0.02

    @pytest.mark.parametrize("preset", MULTIROTOR_PRESETS)
    def test_attitude_commands_do_not_bleed_into_altitude(self, preset: VehiclePreset):
        """A pure torque is pure: the collective must not move with it."""
        aircraft = create_multirotor(preset)
        hover = aircraft.hover_wrench()
        arm = aircraft.rotors[0].arm_length
        for axis in (1, 2, 3):
            wrench = hover.copy()
            wrench[axis] = 0.05 * hover[0] * arm
            total = aircraft.mixer.wrench_to_forces(wrench).sum()
            assert total == pytest.approx(hover[0], rel=1e-9)


class TestYawAuthorityIsEarnedNotAsserted:
    @staticmethod
    def _hex_with_every_rotor_turning_the_same_way() -> Multirotor:
        """The S550 with one thing changed: nothing counter-rotates."""
        params = get_params(VehiclePreset.HEX_S550)
        return Multirotor(
            MultirotorParams(
                mass=params.mass,
                inertia=params.inertia,
                rotors=[Rotor(r.position, direction=1) for r in params.rotors],
                k_thrust=params.k_thrust,
                k_torque=params.k_torque,
                motor_tau=params.motor_tau,
                omega_max=params.omega_max,
                drag_coeff=params.drag_coeff,
                name="hex_all_ccw",
            )
        )

    def test_yaw_torque_is_chained_to_thrust_when_no_rotor_counter_rotates(self):
        """Yaw is the reaction torque of rotor drag, nothing else.

        Same six positions, same arms, same coefficients — only the spin
        pattern changes, and the yaw row of the allocation matrix collapses
        onto the thrust row. Ask for more yaw, less yaw or none at all and
        the airframe delivers the same thing: minus kappa times whatever it
        is lifting with.
        """
        aircraft = self._hex_with_every_rotor_turning_the_same_way()
        assert aircraft.mixer.yaw_authority == pytest.approx(0.0)

        kappa = aircraft.mixer.kappa
        thrust = aircraft.hover_wrench()[0]
        for commanded_yaw in (-0.3, 0.0, 0.3):
            request = np.array([thrust, 0.0, 0.0, commanded_yaw])
            delivered = aircraft.mixer.forces_to_wrench(aircraft.mixer.wrench_to_forces(request))
            assert delivered[3] == pytest.approx(-kappa * delivered[0], abs=1e-12)

    def test_a_same_spin_hex_spins_up_even_with_no_yaw_command(self):
        """Six rotors dragging the same way take the airframe with them."""
        aircraft = self._hex_with_every_rotor_turning_the_same_way()
        aircraft.reset(position=np.array([0.0, 0.0, 5.0]))
        aircraft.spin_up_to_hover()
        for _ in range(300):
            aircraft.step(aircraft.hover_wrench(), DT)

        healthy = _hovering(VehiclePreset.HEX_S550)
        for _ in range(300):
            healthy.step(healthy.hover_wrench(), DT)

        assert abs(aircraft.euler[2]) > 0.1
        assert abs(healthy.euler[2]) < 1e-9

    def test_the_same_airframe_yaws_once_the_pattern_alternates(self):
        healthy = _hovering(VehiclePreset.HEX_S550)
        wrench = healthy.hover_wrench()
        wrench[3] = 0.02 * wrench[0] * 0.275
        for _ in range(300):
            healthy.step(wrench, DT)
        assert healthy.euler[2] > 0.05

    def test_yaw_authority_is_independent_of_arm_length(self):
        """Lengthening the arms buys roll and pitch, never yaw."""
        short = Multirotor(MultirotorParams(rotors=x_layout(6, 0.15)))
        long_ = Multirotor(MultirotorParams(rotors=x_layout(6, 0.60)))
        assert short.mixer.yaw_authority == pytest.approx(long_.mixer.yaw_authority)
        assert long_.mixer.allocation[1].max() > short.mixer.allocation[1].max()


class TestCoaxial:
    def test_the_octo_stacks_eight_rotors_on_four_arms(self):
        octo = create_multirotor(VehiclePreset.OCTO_X8)
        assert octo.n_rotors == 8
        assert len(octo.motors) == 8
        ground_positions = {(round(r.x, 6), round(r.y, 6)) for r in octo.rotors}
        assert len(ground_positions) == 4

    def test_the_lower_rotor_of_a_pair_must_spin_faster_for_the_same_thrust(self):
        """It works in the upper rotor's wake, so its thrust curve is derated."""
        octo = _hovering(VehiclePreset.OCTO_X8)
        thrusts = octo.get_rotor_thrusts()
        speeds = octo.get_motor_speeds()
        np.testing.assert_allclose(thrusts[::2], thrusts[1::2], rtol=1e-9)
        assert np.all(speeds[1::2] > speeds[::2])

    def test_wake_losses_change_the_hover_speed(self):
        """If lower_efficiency did nothing, this airframe would be a lie."""
        base = x_layout(4, 0.35)
        clean = Multirotor(
            MultirotorParams(
                mass=4.5,
                rotors=coaxial_layout(base, lower_efficiency=1.0),
                k_thrust=4.0e-5,
                k_torque=9.6e-7,
                omega_max=620.0,
            )
        )
        realistic = Multirotor(
            MultirotorParams(
                mass=4.5,
                rotors=coaxial_layout(base, lower_efficiency=0.85),
                k_thrust=4.0e-5,
                k_torque=9.6e-7,
                omega_max=620.0,
            )
        )
        clean.spin_up_to_hover()
        realistic.spin_up_to_hover()
        assert realistic.get_motor_speeds().sum() > clean.get_motor_speeds().sum()

    def test_an_octo_has_no_more_roll_authority_than_its_quad(self):
        """Eight rotors on four arms buy redundancy and thrust, not leverage."""
        octo = create_multirotor(VehiclePreset.OCTO_X8)
        octo_roll = np.abs(octo.mixer.allocation[1]).max()
        quad_roll = 0.35 / np.sqrt(2.0)
        assert octo_roll == pytest.approx(quad_roll, rel=1e-9)


class TestPresets:
    @pytest.mark.parametrize("preset", [p for p in VehiclePreset if p is not VehiclePreset.CUSTOM])
    def test_every_preset_flies(self, preset: VehiclePreset):
        aircraft = create_multirotor(preset)
        aircraft.reset(position=np.array([0.0, 0.0, 3.0]))
        aircraft.spin_up_to_hover()
        for _ in range(500):
            aircraft.step(aircraft.hover_wrench(), DT)
        assert np.all(np.isfinite(aircraft.state))
        assert aircraft.mixer.fully_actuated

    def test_the_hex_has_six_rotors(self):
        assert create_multirotor(VehiclePreset.HEX_S550).n_rotors == 6

    def test_the_octo_has_eight(self):
        assert create_multirotor(VehiclePreset.OCTO_X8).n_rotors == 8

    def test_create_multirotor_returns_a_quadrotor_for_a_four_rotor_preset(self):
        assert isinstance(create_multirotor(VehiclePreset.RACING_250), Quadrotor)

    def test_create_quadrotor_refuses_a_hexacopter(self):
        with pytest.raises(ValueError, match="not a quadrotor"):
            create_quadrotor(VehiclePreset.HEX_S550)

    def test_overrides_apply_to_multirotor_presets(self):
        heavy = create_multirotor(VehiclePreset.HEX_S550, mass=3.0)
        assert heavy.params.mass == 3.0
        assert heavy.n_rotors == 6
        assert get_params(VehiclePreset.HEX_S550).mass == 1.8, "the preset must not be mutated"

    def test_an_unknown_override_is_rejected(self):
        with pytest.raises(TypeError, match="wingspan"):
            create_multirotor(VehiclePreset.HEX_S550, wingspan=2.0)

    def test_the_custom_preset_takes_a_layout(self):
        aircraft = create_multirotor(VehiclePreset.CUSTOM, mass=1.0, rotors=x_layout(8, 0.4))
        assert aircraft.n_rotors == 8

    def test_thrust_to_weight_accounts_for_wake_losses(self):
        """An X8 must not be credited with eight clean-air rotors."""
        octo = get_params(VehiclePreset.OCTO_X8)
        naive = 8 * octo.max_rotor_thrust / (octo.mass * octo.gravity)
        assert octo.thrust_to_weight < naive
        assert octo.thrust_to_weight > 2.0


class TestModelPlumbing:
    def test_the_rotor_count_drives_the_motor_count(self):
        for n in (4, 6, 8, 12):
            aircraft = Multirotor(MultirotorParams(rotors=x_layout(n, 0.3)))
            assert len(aircraft.motors) == n
            assert aircraft.get_motor_speeds().shape == (n,)
            assert aircraft.get_rotor_thrusts().shape == (n,)

    def test_reset_stops_every_motor(self):
        aircraft = _hovering(VehiclePreset.HEX_S550)
        assert np.all(aircraft.get_motor_speeds() > 0.0)
        aircraft.reset()
        np.testing.assert_array_equal(aircraft.get_motor_speeds(), np.zeros(6))
        assert aircraft.time == 0.0

    def test_motors_start_stopped_so_a_cold_start_sinks(self):
        aircraft = create_multirotor(VehiclePreset.HEX_S550)
        aircraft.reset(position=np.array([0.0, 0.0, 5.0]))
        for _ in range(100):
            aircraft.step(aircraft.hover_wrench(), DT)
        assert aircraft.position[2] < 5.0

    def test_rotor_positions_and_spins_are_exposed_as_arrays(self):
        aircraft = create_multirotor(VehiclePreset.HEX_S550)
        assert aircraft.rotor_positions.shape == (6, 3)
        np.testing.assert_array_equal(aircraft.spin_directions, [1, -1, 1, -1, 1, -1])

    def test_the_ground_plane_stops_the_aircraft_sinking(self):
        aircraft = create_multirotor(VehiclePreset.HEX_S550)
        aircraft.reset(position=np.array([0.0, 0.0, 0.5]))
        for _ in range(2000):
            aircraft.step(np.zeros(4), DT)
        assert aircraft.position[2] == 0.0

    def test_time_advances_with_the_step(self):
        aircraft = create_multirotor(VehiclePreset.OCTO_X8)
        for _ in range(100):
            aircraft.step(aircraft.hover_wrench(), 0.01)
        assert aircraft.time == pytest.approx(1.0)

    def test_repr_names_the_airframe(self):
        assert "octo_x8" in repr(create_multirotor(VehiclePreset.OCTO_X8))


class TestSaturationInFlight:
    def test_prioritising_torque_keeps_a_starved_hex_upright(self):
        """Half the thrust it needs, and a roll command it cannot afford.

        Clipping spends the roll command paying for the thrust deficit and
        the aircraft rolls the wrong way; prioritising torque keeps the roll
        exact and lets the altitude go, which is the trade a real flight
        stack makes.
        """
        wrench = None
        results = {}
        for strategy in ("clip", "prioritise_torque"):
            aircraft = create_multirotor(VehiclePreset.HEX_S550, saturation=strategy)
            aircraft.reset(position=np.array([0.0, 0.0, 20.0]))
            aircraft.spin_up_to_hover()
            wrench = np.array([0.35 * aircraft.hover_wrench()[0], 1.6, 0.0, 0.0])
            for _ in range(300):
                aircraft.step(wrench, DT)
            results[strategy] = aircraft

        delivered = {
            name: craft.mixer.forces_to_wrench(craft.mixer.wrench_to_forces(wrench))
            for name, craft in results.items()
        }
        assert delivered["prioritise_torque"][1] == pytest.approx(wrench[1], abs=1e-9)
        assert abs(delivered["clip"][1] - wrench[1]) > 0.1
        assert results["prioritise_torque"].euler[0] > results["clip"].euler[0]
