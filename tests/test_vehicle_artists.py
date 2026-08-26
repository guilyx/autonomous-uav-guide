# Erwin Lejeune - 2026-08-26
"""Tests for the VTOL artist and the display-attitude helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from uav_sim.visualization import (
    attitude_from_velocity,
    attitude_series_from_positions,
    clear_vehicle_artists,
    draw_vtol_3d,
)


def _axes3d():
    fig = plt.figure()
    return fig, fig.add_subplot(111, projection="3d")


class TestAttitudeFromVelocity:
    def test_level_flight_is_identity(self):
        R = attitude_from_velocity(np.array([5.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_heading_follows_velocity(self):
        R = attitude_from_velocity(np.array([0.0, 5.0, 0.0]))
        np.testing.assert_allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_climb_raises_the_nose(self):
        nose = attitude_from_velocity(np.array([5.0, 0.0, 5.0]))[:, 0]
        assert nose[2] == pytest.approx(np.sqrt(0.5), abs=1e-9)

    def test_descent_lowers_the_nose(self):
        nose = attitude_from_velocity(np.array([5.0, 0.0, -5.0]))[:, 0]
        assert nose[2] == pytest.approx(-np.sqrt(0.5), abs=1e-9)

    def test_left_turn_drops_the_left_wing(self):
        # Flying +x, accelerating towards +y: a left turn.
        left_wing = attitude_from_velocity(
            np.array([10.0, 0.0, 0.0]), np.array([0.0, 9.81, 0.0])
        )[:, 1]
        assert left_wing[2] < 0.0

    def test_right_turn_raises_the_left_wing(self):
        left_wing = attitude_from_velocity(
            np.array([10.0, 0.0, 0.0]), np.array([0.0, -9.81, 0.0])
        )[:, 1]
        assert left_wing[2] > 0.0

    def test_bank_is_limited(self):
        # An absurd lateral acceleration must not roll past the limit.
        R = attitude_from_velocity(
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, 1e6, 0.0]),
            max_bank=np.radians(30.0),
        )
        roll = np.arctan2(R[2, 1], R[2, 2])
        assert abs(roll) <= np.radians(30.0) + 1e-9

    def test_straight_line_acceleration_does_not_bank(self):
        R = attitude_from_velocity(np.array([10.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_matrix_is_orthonormal(self):
        R = attitude_from_velocity(np.array([3.0, -4.0, 2.0]), np.array([1.0, 2.0, -0.5]))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "velocity",
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([np.nan, 0.0, 0.0]),
            np.array([1e-9, 0.0, 0.0]),
        ],
    )
    def test_degenerate_velocity_falls_back_to_identity(self, velocity):
        np.testing.assert_allclose(attitude_from_velocity(velocity), np.eye(3))


class TestAttitudeSeries:
    def test_one_matrix_per_sample(self):
        pos = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
        assert len(attitude_series_from_positions(pos, 0.1)) == 10

    def test_straight_line_stays_level(self):
        pos = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
        for R in attitude_series_from_positions(pos, 0.1):
            np.testing.assert_allclose(R, np.eye(3), atol=1e-9)

    def test_circle_banks_consistently(self):
        t = np.linspace(0.0, 2.0 * np.pi, 200)
        pos = np.column_stack([np.cos(t) * 10.0, np.sin(t) * 10.0, np.zeros_like(t)])
        # Counter-clockwise is a continuous left turn, so the left wing is
        # low for the whole lap rather than flapping sign.
        signs = [R[2, 1] < 0.0 for R in attitude_series_from_positions(pos, t[1] - t[0])[5:-5]]
        assert all(signs)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            attitude_series_from_positions(np.zeros((5, 2)), 0.1)

    def test_degenerate_inputs_are_identity(self):
        assert attitude_series_from_positions(np.zeros((1, 3)), 0.1) == [
            pytest.approx(np.eye(3))
        ]
        for R in attitude_series_from_positions(np.zeros((4, 3)), 0.0):
            np.testing.assert_allclose(R, np.eye(3))


class TestDrawVtol3d:
    @pytest.mark.parametrize("tilt", [0.0, np.pi / 4, np.pi / 2])
    def test_returns_removable_artists(self, tilt):
        fig, ax = _axes3d()
        arts = draw_vtol_3d(ax, np.zeros(3), np.eye(3), tilt=tilt)
        assert len(arts) > 0
        clear_vehicle_artists(arts)
        assert arts == []
        plt.close(fig)

    def test_tilt_moves_the_rotor_axis(self):
        """Hover points the nacelles up, cruise points them forward."""
        fig, ax = _axes3d()

        def rotor_tips(tilt):
            arts = draw_vtol_3d(ax, np.zeros(3), np.eye(3), tilt=tilt)
            # Nacelle lines are the ones drawn in the rotor colour.
            tips = [
                a.get_data_3d()
                for a in arts
                if hasattr(a, "get_color") and a.get_color() == "orangered"
            ]
            clear_vehicle_artists(arts)
            return tips

        hover = rotor_tips(0.0)
        cruise = rotor_tips(np.pi / 2)
        assert hover and cruise
        # In hover the nacelle rises in z; in cruise it extends in x instead.
        hover_dz = max(abs(z[1] - z[0]) for _, _, z in hover)
        cruise_dz = max(abs(z[1] - z[0]) for _, _, z in cruise)
        cruise_dx = max(abs(x[1] - x[0]) for x, _, _ in cruise)
        assert hover_dz > cruise_dz
        assert cruise_dx > cruise_dz
        plt.close(fig)

    def test_position_offsets_the_model(self):
        fig, ax = _axes3d()
        origin = draw_vtol_3d(ax, np.zeros(3), np.eye(3))
        xs_origin = [a.get_data_3d()[0] for a in origin if hasattr(a, "get_data_3d")]
        clear_vehicle_artists(origin)
        moved = draw_vtol_3d(ax, np.array([100.0, 0.0, 0.0]), np.eye(3))
        xs_moved = [a.get_data_3d()[0] for a in moved if hasattr(a, "get_data_3d")]
        clear_vehicle_artists(moved)
        assert min(map(min, xs_moved)) > max(map(max, xs_origin))
        plt.close(fig)
