# Erwin Lejeune - 2026-08-27
"""Tests for the control barrier functions and the QP that enforces them."""

import numpy as np
import pytest

from flybots.safety import (
    AltitudeFloorBarrier,
    ConnectivityBarrier,
    GeofenceBoxBarrier,
    InfeasibleQPError,
    SafeDistanceBarrier,
    SafetyFilter,
    SpeedLimitBarrier,
    SphereObstacleBarrier,
    solve_safety_qp,
)


class TestSafetyQP:
    def test_no_constraints_passes_through(self):
        u = solve_safety_qp(np.array([1.0, 2.0, 3.0]), None, None)
        np.testing.assert_allclose(u, [1.0, 2.0, 3.0])

    def test_inactive_constraint_leaves_command_alone(self):
        u = solve_safety_qp(np.zeros(2), np.array([[1.0, 0.0]]), np.array([5.0]))
        np.testing.assert_allclose(u, [0.0, 0.0])

    def test_active_constraint_projects_onto_the_face(self):
        u = solve_safety_qp(np.array([3.0, 0.0]), np.array([[1.0, 0.0]]), np.array([1.0]))
        np.testing.assert_allclose(u, [1.0, 0.0], atol=1e-9)

    def test_correction_is_minimal(self):
        """Only the violated direction may move; the rest is untouched."""
        u = solve_safety_qp(np.array([3.0, 7.0]), np.array([[1.0, 0.0]]), np.array([1.0]))
        np.testing.assert_allclose(u, [1.0, 7.0], atol=1e-9)

    def test_oblique_constraint(self):
        u = solve_safety_qp(np.array([2.0, 2.0]), np.array([[1.0, 1.0]]), np.array([1.0]))
        np.testing.assert_allclose(u, [0.5, 0.5], atol=1e-9)

    def test_actuator_bounds_are_constraints(self):
        u = solve_safety_qp(np.array([10.0, -10.0]), None, None, u_min=-2.0, u_max=3.0)
        np.testing.assert_allclose(u, [3.0, -2.0], atol=1e-9)

    def test_contradictory_constraints_raise(self):
        with pytest.raises(InfeasibleQPError):
            solve_safety_qp(np.zeros(1), np.array([[1.0], [-1.0]]), np.array([0.0, -1.0]))

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            solve_safety_qp(np.zeros(2), np.array([[1.0, 0.0, 0.0]]), np.array([1.0]))


class TestBarrierSigns:
    """A barrier must be positive when safe and negative when not."""

    def test_safe_distance(self):
        b = SafeDistanceBarrier(2.0)
        far = np.array([[0.0, 0, 0], [5.0, 0, 0]])
        near = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        v = np.zeros((2, 3))
        assert b.margin(far, v) > 0
        assert b.margin(near, v) < 0

    def test_connectivity_is_the_mirror_image(self):
        b = ConnectivityBarrier(3.0)
        v = np.zeros((2, 3))
        assert b.margin(np.array([[0.0, 0, 0], [1.0, 0, 0]]), v) > 0
        assert b.margin(np.array([[0.0, 0, 0], [9.0, 0, 0]]), v) < 0

    def test_sphere_obstacle(self):
        b = SphereObstacleBarrier(np.array([[0.0, 0, 0]]), 2.0)
        v = np.zeros((1, 3))
        assert b.margin(np.array([[5.0, 0, 0]]), v) > 0
        assert b.margin(np.array([[1.0, 0, 0]]), v) < 0

    def test_geofence(self):
        b = GeofenceBoxBarrier(np.zeros(3), np.full(3, 10.0))
        v = np.zeros((1, 3))
        assert b.margin(np.array([[5.0, 5, 5]]), v) > 0
        assert b.margin(np.array([[11.0, 5, 5]]), v) < 0

    def test_altitude_floor(self):
        b = AltitudeFloorBarrier(2.0)
        v = np.zeros((1, 3))
        assert b.margin(np.array([[0.0, 0, 5]]), v) > 0
        assert b.margin(np.array([[0.0, 0, 1]]), v) < 0

    def test_speed_limit(self):
        b = SpeedLimitBarrier(5.0)
        p = np.zeros((1, 3))
        assert b.margin(p, np.array([[1.0, 0, 0]])) > 0
        assert b.margin(p, np.array([[9.0, 0, 0]])) < 0


class TestBarrierValidation:
    @pytest.mark.parametrize(
        "factory,match",
        [
            (lambda: SafeDistanceBarrier(-1.0), "safe_distance must be positive"),
            (lambda: SafeDistanceBarrier(1.0, k1=0.0), "class-K gains must be positive"),
            (lambda: ConnectivityBarrier(0.0), "comm_range must be positive"),
            (lambda: SpeedLimitBarrier(0.0), "max_speed must be positive"),
            (
                lambda: GeofenceBoxBarrier(np.ones(3), np.zeros(3)),
                "upper must exceed lower",
            ),
        ],
    )
    def test_rejects_nonsense(self, factory, match):
        with pytest.raises(ValueError, match=match):
            factory()


class TestRelativeDegree:
    """Position barriers must actually constrain acceleration."""

    def test_position_barrier_constrains_input_when_closing(self):
        # Two vehicles closing head-on: the row must be non-zero, otherwise
        # the high-order treatment collapsed and the filter is a no-op.
        b = SafeDistanceBarrier(2.0)
        p = np.array([[0.0, 0, 0], [3.0, 0, 0]])
        v = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
        rows = b.rows(p, v)
        assert rows.A.shape == (1, 6)
        assert np.linalg.norm(rows.A) > 0

    def test_speed_limit_row_vanishes_at_rest(self):
        """At zero velocity no acceleration can instantaneously overspeed."""
        rows = SpeedLimitBarrier(5.0).rows(np.zeros((1, 3)), np.zeros((1, 3)))
        np.testing.assert_allclose(rows.A, np.zeros((1, 3)))


class TestSafetyFilter:
    def test_leaves_a_safe_command_untouched(self):
        f = SafetyFilter([SafeDistanceBarrier(1.0)])
        p = np.array([[0.0, 0, 0], [50.0, 0, 0]])
        v = np.zeros((2, 3))
        nominal = np.array([[1.0, 0, 0], [0.0, 1, 0]])
        rep = f(p, v, nominal)
        np.testing.assert_allclose(rep.command, nominal, atol=1e-9)
        assert not rep.intervened

    def test_intervenes_on_a_collision_course(self):
        f = SafetyFilter([SafeDistanceBarrier(2.0, k1=5.0, k2=5.0)])
        p = np.array([[0.0, 0, 0], [2.5, 0, 0]])
        v = np.array([[3.0, 0, 0], [-3.0, 0, 0]])
        nominal = np.array([[5.0, 0, 0], [-5.0, 0, 0]])  # straight at each other
        rep = f(p, v, nominal)
        assert rep.intervened
        assert rep.correction_norm > 0.0

    def test_pair_constraint_shares_the_effort(self):
        """A fleet-wide QP moves both vehicles, not just one."""
        f = SafetyFilter([SafeDistanceBarrier(2.0, k1=5.0, k2=5.0)])
        p = np.array([[0.0, 0, 0], [2.5, 0, 0]])
        v = np.array([[3.0, 0, 0], [-3.0, 0, 0]])
        rep = f(p, v, np.array([[5.0, 0, 0], [-5.0, 0, 0]]))
        delta = rep.command - np.array([[5.0, 0, 0], [-5.0, 0, 0]])
        assert np.linalg.norm(delta[0]) > 1e-6
        assert np.linalg.norm(delta[1]) > 1e-6

    def test_respects_actuator_limits(self):
        f = SafetyFilter([SafeDistanceBarrier(1.0)], u_min=-2.0, u_max=2.0)
        rep = f(
            np.array([[0.0, 0, 0], [50.0, 0, 0]]),
            np.zeros((2, 3)),
            np.array([[99.0, 0, 0], [-99.0, 0, 0]]),
        )
        assert np.all(rep.command <= 2.0 + 1e-9)
        assert np.all(rep.command >= -2.0 - 1e-9)

    def test_impossible_barriers_report_infeasible(self):
        """Safe distance beyond comms range cannot both hold."""
        f = SafetyFilter(
            [SafeDistanceBarrier(20.0, k1=8.0, k2=8.0), ConnectivityBarrier(3.0, k1=8.0, k2=8.0)],
            u_max=1.0,
            u_min=-1.0,
        )
        rep = f(
            np.array([[0.0, 0, 0], [10.0, 0, 0]]),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
        )
        assert rep.infeasible

    def test_fallback_raise_propagates(self):
        f = SafetyFilter(
            [SafeDistanceBarrier(20.0, k1=8.0, k2=8.0), ConnectivityBarrier(3.0, k1=8.0, k2=8.0)],
            u_max=1.0,
            u_min=-1.0,
            fallback="raise",
        )
        with pytest.raises(InfeasibleQPError):
            f(np.array([[0.0, 0, 0], [10.0, 0, 0]]), np.zeros((2, 3)), np.zeros((2, 3)))

    def test_rejects_bad_fallback(self):
        with pytest.raises(ValueError, match="fallback must be"):
            SafetyFilter([], fallback="explode")

    def test_rejects_shape_mismatch(self):
        f = SafetyFilter([SafeDistanceBarrier(1.0)])
        with pytest.raises(ValueError, match="must agree"):
            f(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((2, 3)))


class TestClosedLoopSafety:
    """The property that matters: simulate, and check nothing is violated."""

    @staticmethod
    def _swap(n, d_safe, use_filter, steps=1600, dt=0.02):
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        p = np.column_stack([12.0 * np.cos(ang), 12.0 * np.sin(ang), np.full(n, 20.0)])
        goal = -p.copy()
        v = np.zeros_like(p)
        filt = SafetyFilter(
            [SafeDistanceBarrier(d_safe, k1=6.0, k2=6.0), SpeedLimitBarrier(6.0)],
            u_max=8.0,
            u_min=-8.0,
        )
        worst = np.inf
        for _ in range(steps):
            u = np.clip(3.0 * (goal - p) - 3.5 * v, -8.0, 8.0)
            if use_filter:
                u = filt(p, v, u).command
            v = v + u * dt
            p = p + v * dt
            worst = min(
                worst,
                min(float(np.linalg.norm(p[i] - p[j])) for i in range(n) for j in range(i + 1, n)),
            )
        return worst, p, goal

    def test_unfiltered_swap_collides(self):
        """Without the filter the ring passes through itself."""
        worst, _, _ = self._swap(8, 1.6, use_filter=False)
        assert worst < 0.5

    def test_filtered_swap_never_violates(self):
        # 2% tolerance for the forward-Euler step: the barrier holds in
        # continuous time, and discretisation lets it overshoot slightly.
        worst, _, _ = self._swap(8, 1.6, use_filter=True)
        assert worst >= 1.6 * 0.98

    def test_filtered_swap_still_reaches_its_goal(self):
        """Safety must not be bought by simply refusing to move."""
        _, p, goal = self._swap(8, 1.6, use_filter=True, steps=2500)
        assert float(np.mean(np.linalg.norm(p - goal, axis=1))) < 0.5

    def test_geofence_contains_a_fleeing_vehicle(self):
        lower, upper = np.zeros(3), np.full(3, 20.0)
        filt = SafetyFilter(
            [GeofenceBoxBarrier(lower, upper, k1=6.0, k2=6.0)], u_max=6.0, u_min=-6.0
        )
        p = np.array([[10.0, 10.0, 10.0]])
        v = np.zeros((1, 3))
        for _ in range(1500):
            u = filt(p, v, np.array([[6.0, 0.0, 0.0]])).command  # flat out at the wall
            v = v + u * 0.02
            p = p + v * 0.02
        assert p[0, 0] <= upper[0] + 0.2
