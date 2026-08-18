# Erwin Lejeune - 2026-08-18
"""Tests for the geometry-derived control allocation.

The load-bearing test in this file is the first one: the 4x4 mixing
matrices this library shipped for two years, written out by hand, are
reproduced from rotor positions and spin directions alone. Everything the
hexacopter and the octocopter do rests on that agreement.
"""

import numpy as np
import pytest

from uav_sim.vehicles.components.allocation import (
    ControlAllocation,
    Rotor,
    allocation_matrix,
    coaxial_layout,
    h_layout,
    plus_layout,
    radial_layout,
    x_layout,
)
from uav_sim.vehicles.components.mixer import Mixer

ARM = 0.175
K_THRUST = 8.55e-6
K_TORQUE = 1.36e-7
KAPPA = K_TORQUE / K_THRUST


def _historical_x_mixer() -> np.ndarray:
    """The X-frame matrix as it was hard-coded before it was derived.

    Kept as a literal on purpose. Deriving the reference from the same
    geometry as the code under test would assert nothing.
    """
    s = ARM / np.sqrt(2.0)
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [s, -s, -s, s],
            [s, s, -s, -s],
            [-KAPPA, KAPPA, -KAPPA, KAPPA],
        ]
    )


def _historical_plus_mixer() -> np.ndarray:
    """The +-frame matrix as it was hard-coded before it was derived."""
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -ARM, 0.0, ARM],
            [ARM, 0.0, -ARM, 0.0],
            [-KAPPA, KAPPA, -KAPPA, KAPPA],
        ]
    )


class TestDerivationMatchesTheHardCodedMixer:
    """The whole change stands or falls here."""

    def test_x_frame_matrix_is_reproduced(self):
        derived = Mixer(arm_length=ARM, k_thrust=K_THRUST, k_torque=K_TORQUE, frame="x")
        np.testing.assert_allclose(derived.mix_matrix, _historical_x_mixer(), atol=1e-15)

    def test_plus_frame_matrix_is_reproduced(self):
        derived = Mixer(arm_length=ARM, k_thrust=K_THRUST, k_torque=K_TORQUE, frame="+")
        np.testing.assert_allclose(derived.mix_matrix, _historical_plus_mixer(), atol=1e-15)

    def test_x_frame_inverse_is_reproduced(self):
        derived = Mixer(arm_length=ARM, k_thrust=K_THRUST, k_torque=K_TORQUE, frame="x")
        expected = np.linalg.inv(_historical_x_mixer())
        np.testing.assert_allclose(derived.inv_mix_matrix, expected, atol=1e-12)

    def test_the_derived_column_order_is_rear_left_first(self):
        """The old comment said FL, FR, RR, RL. The matrix says otherwise.

        Only one assignment of positions and spins reproduces all four rows,
        and it starts at the rear left. Getting this backwards would put
        thrust on the wrong pair of motors for every pitch command.
        """
        rotors = Mixer.frame_layout("x", ARM)
        s = ARM / np.sqrt(2.0)
        expected = [(-s, s), (-s, -s), (s, -s), (s, s)]
        for rotor, (x, y) in zip(rotors, expected):
            assert rotor.x == pytest.approx(x, abs=1e-12)
            assert rotor.y == pytest.approx(y, abs=1e-12)
        assert [r.direction for r in rotors] == [1, -1, 1, -1]

    def test_plus_frame_puts_a_rotor_exactly_on_each_axis(self):
        rotors = Mixer.frame_layout("+", ARM)
        assert [(r.x, r.y) for r in rotors] == [
            (-ARM, 0.0),
            (0.0, -ARM),
            (ARM, 0.0),
            (0.0, ARM),
        ]


class TestAllocationSigns:
    """The rows follow from FLU, and FLU is not negotiable."""

    def test_thrust_row_is_all_ones(self):
        A = allocation_matrix(x_layout(6, 0.3), KAPPA)
        np.testing.assert_allclose(A[0], np.ones(6))

    def test_roll_torque_comes_from_the_left_hand_rotors(self):
        """tau_x = sum(y_i f_i): pushing up on the left rolls right (phi > 0)."""
        rotors = x_layout(6, 0.3)
        A = allocation_matrix(rotors, KAPPA)
        for column, rotor in zip(A.T, rotors):
            assert column[1] == pytest.approx(rotor.y)

    def test_a_front_rotor_produces_a_nose_up_pitch_torque(self):
        """In FLU positive theta is nose-down, so a front rotor gives tau_y < 0."""
        front = Rotor(np.array([0.3, 0.0, 0.0]), direction=1)
        rear = Rotor(np.array([-0.3, 0.0, 0.0]), direction=-1)
        A = allocation_matrix([front, rear], KAPPA)
        assert A[2, 0] < 0.0
        assert A[2, 1] > 0.0

    def test_a_ccw_rotor_drags_the_airframe_clockwise(self):
        """Reaction torque opposes the rotor: sigma = +1 gives tau_z < 0."""
        ccw = Rotor(np.array([0.3, 0.0, 0.0]), direction=1)
        cw = Rotor(np.array([-0.3, 0.0, 0.0]), direction=-1)
        A = allocation_matrix([ccw, cw], KAPPA)
        assert A[3, 0] == pytest.approx(-KAPPA)
        assert A[3, 1] == pytest.approx(KAPPA)

    def test_vertical_offset_does_not_enter_the_allocation(self):
        """A rotor thrusting along +z has no moment arm about z."""
        low = Rotor(np.array([0.3, 0.1, -0.4]), direction=1)
        high = Rotor(np.array([0.3, 0.1, 0.9]), direction=1)
        np.testing.assert_allclose(
            allocation_matrix([low], KAPPA), allocation_matrix([high], KAPPA)
        )


class TestRotor:
    def test_direction_must_be_plus_or_minus_one(self):
        with pytest.raises(ValueError, match="direction"):
            Rotor(np.zeros(3), direction=0)

    def test_thrust_scale_must_be_positive(self):
        with pytest.raises(ValueError, match="thrust_scale"):
            Rotor(np.zeros(3), thrust_scale=0.0)

    def test_arm_length_ignores_the_vertical_offset(self):
        rotor = Rotor(np.array([0.3, 0.4, 5.0]))
        assert rotor.arm_length == pytest.approx(0.5)


class TestLayouts:
    def test_radial_rejects_an_odd_rotor_count(self):
        with pytest.raises(ValueError, match="even number"):
            radial_layout(5, 0.3)

    def test_radial_rejects_fewer_than_four(self):
        with pytest.raises(ValueError, match="even number"):
            radial_layout(2, 0.3)

    def test_radial_alternates_spin_all_the_way_round(self):
        rotors = radial_layout(8, 0.3)
        assert sum(r.direction for r in rotors) == 0
        for a, b in zip(rotors, rotors[1:]):
            assert a.direction == -b.direction

    def test_every_rotor_sits_at_the_arm_length(self):
        for rotor in x_layout(6, 0.275):
            assert rotor.arm_length == pytest.approx(0.275)

    def test_x_frame_is_the_plus_frame_rotated_half_a_sector(self):
        n = 6
        x_angles = sorted(np.arctan2(r.y, r.x) % (2 * np.pi) for r in x_layout(n, 0.3))
        plus_angles = sorted(np.arctan2(r.y, r.x) % (2 * np.pi) for r in plus_layout(n, 0.3))
        offsets = [(a - b) % (2 * np.pi) for a, b in zip(x_angles, plus_angles)]
        np.testing.assert_allclose(offsets, np.pi / n, atol=1e-12)

    def test_hex_x_straddles_the_forward_axis_and_hex_plus_sits_on_it(self):
        on_the_nose = [r for r in x_layout(6, 0.3) if r.y == 0.0 and r.x > 0]
        assert on_the_nose == []
        assert [(r.x, r.y) for r in plus_layout(6, 0.3) if r.y == 0.0 and r.x > 0] == [(0.3, 0.0)]

    def test_h_layout_separates_length_from_width(self):
        rotors = h_layout(4, length=0.40, width=0.24)
        xs = sorted({round(r.x, 9) for r in rotors})
        ys = sorted({round(r.y, 9) for r in rotors})
        assert xs == [-0.20, 0.20]
        assert ys == [-0.12, 0.12]

    def test_h_layout_with_six_rotors_has_three_rows(self):
        rotors = h_layout(6, length=0.40, width=0.24)
        assert len({round(r.x, 9) for r in rotors}) == 3
        assert sum(r.direction for r in rotors) == 0

    def test_h_layout_degenerates_to_the_x_frame(self):
        """A square H *is* an X: same rotors, same spins, so same matrix.

        Only the column order differs, so both matrices are sorted by rotor
        position before comparing — that keeps each column's four entries
        together, which is what makes this a claim about the airframe rather
        than about four unrelated rows.
        """
        side = 2 * ARM / np.sqrt(2.0)

        def sorted_columns(rotors):
            order = sorted(range(len(rotors)), key=lambda i: (round(rotors[i].x, 9),
                                                              round(rotors[i].y, 9)))
            return allocation_matrix(rotors, KAPPA)[:, order]

        np.testing.assert_allclose(
            sorted_columns(h_layout(4, length=side, width=side)),
            sorted_columns(x_layout(4, ARM)),
            atol=1e-15,
        )

    def test_h_layout_rejects_an_odd_rotor_count(self):
        with pytest.raises(ValueError, match="even number"):
            h_layout(5)


class TestCoaxialLayout:
    def test_stacking_doubles_the_rotor_count(self):
        assert len(coaxial_layout(x_layout(4, 0.35))) == 8

    def test_a_pair_shares_one_ground_position(self):
        for upper, lower in zip(*[iter(coaxial_layout(x_layout(4, 0.35)))] * 2):
            assert upper.x == pytest.approx(lower.x)
            assert upper.y == pytest.approx(lower.y)
            assert upper.z > lower.z

    def test_a_pair_counter_rotates(self):
        rotors = coaxial_layout(x_layout(4, 0.35))
        for upper, lower in zip(rotors[::2], rotors[1::2]):
            assert upper.direction == -lower.direction

    def test_the_lower_rotor_works_in_a_wake(self):
        rotors = coaxial_layout(x_layout(4, 0.35), lower_efficiency=0.85)
        assert rotors[0].thrust_scale == pytest.approx(1.0)
        assert rotors[1].thrust_scale == pytest.approx(0.85)

    def test_a_coaxial_pair_adds_thrust_and_yaw_but_no_roll_or_pitch(self):
        """The pair's two columns differ in the yaw row alone."""
        rotors = coaxial_layout(x_layout(4, 0.35))
        A = allocation_matrix(rotors, KAPPA)
        upper, lower = A[:, 0], A[:, 1]
        np.testing.assert_allclose(upper[:3], lower[:3], atol=1e-15)
        assert upper[3] == pytest.approx(-lower[3])

    def test_a_coaxial_octo_is_still_fully_actuated(self):
        octo = ControlAllocation(coaxial_layout(x_layout(4, 0.35)), K_THRUST, K_TORQUE)
        assert octo.rank == 4
        assert octo.fully_actuated
        assert octo.unreachable_axes == ()

    def test_a_collinear_coaxial_layout_loses_pitch(self):
        """Four rotors, two positions, both on the lateral axis: no pitch.

        This is the rank-deficient case the pseudo-inverse has to survive.
        A plain inverse would raise; the pseudo-inverse gives the closest
        reachable wrench and simply ignores the pitch it cannot make.
        """
        pair = [
            Rotor(np.array([0.0, 0.3, 0.0]), 1),
            Rotor(np.array([0.0, -0.3, 0.0]), -1),
        ]
        allocation = ControlAllocation(coaxial_layout(pair), K_THRUST, K_TORQUE)
        assert allocation.rank == 3
        assert not allocation.fully_actuated
        assert allocation.unreachable_axes == ("pitch",)

        forces = allocation.wrench_to_forces(np.array([10.0, 0.2, 0.5, 0.0]))
        achieved = allocation.forces_to_wrench(forces)
        assert np.all(np.isfinite(forces))
        assert achieved[2] == pytest.approx(0.0, abs=1e-12)
        assert achieved[1] == pytest.approx(0.2, abs=1e-9)


class TestYawAuthorityComesFromTheSpinPattern:
    def test_an_alternating_pattern_yields_kappa(self):
        """Yaw comes from rotor drag, so it scales with kappa, not the arm."""
        for n in (4, 6, 8):
            for arm in (0.1, 0.5):
                allocation = ControlAllocation(x_layout(n, arm), K_THRUST, K_TORQUE)
                assert allocation.yaw_authority == pytest.approx(KAPPA)

    def test_all_rotors_spinning_the_same_way_have_no_yaw_authority(self):
        rotors = [Rotor(r.position, direction=1) for r in x_layout(6, 0.275)]
        allocation = ControlAllocation(rotors, K_THRUST, K_TORQUE)

        assert allocation.yaw_authority == pytest.approx(0.0)
        assert allocation.rank == 3
        # Thrust goes too: every newton it lifts drags a fixed reaction
        # torque along with it, so neither axis is available on its own.
        assert set(allocation.unreachable_axes) == {"thrust", "yaw"}

    def test_a_same_spin_airframe_cannot_yaw_without_changing_thrust(self):
        rotors = [Rotor(r.position, direction=1) for r in x_layout(6, 0.275)]
        allocation = ControlAllocation(rotors, K_THRUST, K_TORQUE)
        forces = allocation.wrench_to_forces(np.array([18.0, 0.0, 0.0, 0.3]))
        achieved = allocation.forces_to_wrench(forces)
        assert achieved[3] != pytest.approx(0.3, abs=1e-3)

    def test_reversing_every_spin_reverses_the_yaw_row(self):
        forward = allocation_matrix(x_layout(6, 0.275), KAPPA)
        reversed_ = allocation_matrix(
            radial_layout(6, 0.275, offset=np.pi - np.pi / 6, first_direction=-1), KAPPA
        )
        np.testing.assert_allclose(forward[:3], reversed_[:3], atol=1e-15)
        np.testing.assert_allclose(forward[3], -reversed_[3], atol=1e-15)


class TestRedundantAllocation:
    """More rotors than degrees of freedom: the over-determined case."""

    @pytest.fixture
    def hexa(self) -> ControlAllocation:
        return ControlAllocation(x_layout(6, 0.275), K_THRUST, K_TORQUE)

    def test_the_wrench_round_trips_when_nothing_saturates(self, hexa: ControlAllocation):
        wrench = np.array([18.0, 0.15, 0.10, 0.02])
        forces = hexa.wrench_to_forces(wrench)
        assert np.all(forces > 0.0), "test assumes no clamping"
        np.testing.assert_allclose(hexa.forces_to_wrench(forces), wrench, atol=1e-10)

    def test_the_pseudo_inverse_picks_the_smallest_thrust_vector(self, hexa: ControlAllocation):
        """Any other solution with the same wrench costs more rotor effort."""
        wrench = np.array([18.0, 0.15, 0.10, 0.02])
        forces = hexa.wrench_to_forces(wrench)

        null_space = np.linalg.svd(hexa.allocation)[2][4:]
        assert null_space.shape[0] == 2, "a 6-rotor layout has a 2-D null space"
        for direction in null_space:
            for scale in (-0.5, 0.5):
                alternative = forces + scale * direction
                np.testing.assert_allclose(
                    hexa.forces_to_wrench(alternative), wrench, atol=1e-10
                )
                assert np.linalg.norm(alternative) > np.linalg.norm(forces)

    def test_hover_splits_evenly_across_a_symmetric_ring(self, hexa: ControlAllocation):
        forces = hexa.hover_forces(18.0)
        np.testing.assert_allclose(forces, 3.0, atol=1e-10)

    def test_losing_a_rotor_costs_less_on_eight_than_on_four(self):
        """Redundancy is the point of an X8: each rotor carries less load."""
        quad = ControlAllocation(x_layout(4, 0.35), K_THRUST, K_TORQUE)
        octo = ControlAllocation(coaxial_layout(x_layout(4, 0.35)), K_THRUST, K_TORQUE)
        assert octo.hover_forces(44.0).max() < quad.hover_forces(44.0).max()


class TestSaturation:
    """What happens when the solution asks a rotor to pull."""

    MAX = 8.84

    def _allocation(self, saturation: str) -> ControlAllocation:
        return ControlAllocation(
            x_layout(6, 0.275),
            K_THRUST,
            K_TORQUE,
            max_thrust=self.MAX,
            saturation=saturation,
        )

    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="saturation strategy"):
            ControlAllocation(x_layout(4, 0.3), saturation="hope")

    def test_forces_are_always_feasible(self):
        for strategy in ("clip", "prioritise_torque"):
            allocation = self._allocation(strategy)
            for wrench in (
                np.array([6.0, 1.2, 0.4, 0.05]),
                np.array([52.0, 2.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0]),
            ):
                forces = allocation.wrench_to_forces(wrench)
                assert np.all(forces >= 0.0)
                assert np.all(forces <= self.MAX + 1e-12)

    def test_clipping_corrupts_the_torques(self):
        """The default is cheap and wrong in the way that matters."""
        allocation = self._allocation("clip")
        wrench = np.array([6.0, 1.2, 0.4, 0.05])
        achieved = allocation.forces_to_wrench(allocation.wrench_to_forces(wrench))
        assert allocation.last_saturation.clipped
        assert abs(achieved[1] - wrench[1]) > 0.1

    def test_prioritising_torque_keeps_all_three_torques_exactly(self):
        allocation = self._allocation("prioritise_torque")
        for wrench in (np.array([6.0, 1.2, 0.4, 0.05]), np.array([52.0, 2.0, 0.3, 0.1])):
            achieved = allocation.forces_to_wrench(allocation.wrench_to_forces(wrench))
            np.testing.assert_allclose(achieved[1:], wrench[1:], atol=1e-9)
            assert allocation.last_saturation.collective_shift != 0.0

    def test_the_collective_absorbs_the_whole_error(self):
        allocation = self._allocation("prioritise_torque")
        wrench = np.array([6.0, 1.2, 0.4, 0.05])
        achieved = allocation.forces_to_wrench(allocation.wrench_to_forces(wrench))
        assert achieved[0] > wrench[0], "it had to lift harder to keep the roll"

    def test_a_shift_is_only_a_collective_change_on_a_balanced_layout(self):
        balanced = ControlAllocation(x_layout(6, 0.275), K_THRUST, K_TORQUE)
        assert balanced.torque_rows_balanced

        lopsided = ControlAllocation(
            [Rotor(r.position, 1 if i < 4 else -1) for i, r in enumerate(x_layout(6, 0.275))],
            K_THRUST,
            K_TORQUE,
        )
        assert not lopsided.torque_rows_balanced

    def test_a_feasible_request_is_left_exactly_where_the_solver_put_it(self):
        clip = self._allocation("clip")
        prioritise = self._allocation("prioritise_torque")
        wrench = np.array([18.0, 0.15, 0.10, 0.02])
        np.testing.assert_array_equal(
            clip.wrench_to_forces(wrench), prioritise.wrench_to_forces(wrench)
        )
        assert prioritise.last_saturation.collective_shift == 0.0
        assert not prioritise.last_saturation.clipped

    def test_a_request_wider_than_the_band_falls_back_to_clipping(self):
        """No single shift fits a spread larger than the feasible range."""
        allocation = self._allocation("prioritise_torque")
        forces = allocation.wrench_to_forces(np.array([26.0, 40.0, 0.0, 0.0]))
        assert allocation.last_saturation.clipped
        assert allocation.last_saturation.collective_shift == 0.0
        assert np.all(forces >= 0.0)
        assert np.all(forces <= self.MAX + 1e-12)

    def test_the_report_records_what_was_asked_and_what_was_delivered(self):
        allocation = self._allocation("clip")
        wrench = np.array([6.0, 1.2, 0.4, 0.05])
        allocation.wrench_to_forces(wrench)
        report = allocation.last_saturation
        np.testing.assert_array_equal(report.requested, wrench)
        np.testing.assert_allclose(report.error, report.achieved - wrench)


class TestValidation:
    def test_an_empty_layout_is_rejected(self):
        with pytest.raises(ValueError, match="at least one rotor"):
            allocation_matrix([], KAPPA)

    def test_a_negative_thrust_coefficient_is_rejected(self):
        with pytest.raises(ValueError, match="k_thrust"):
            ControlAllocation(x_layout(4, 0.3), k_thrust=0.0)

    def test_a_wrong_length_force_vector_is_rejected(self):
        allocation = ControlAllocation(x_layout(6, 0.3), K_THRUST, K_TORQUE)
        with pytest.raises(ValueError, match="6 rotor forces"):
            allocation.forces_to_wrench(np.ones(4))

    def test_repr_shows_the_spin_pattern(self):
        assert "spins='+-+-+-'" in repr(ControlAllocation(x_layout(6, 0.3)))
