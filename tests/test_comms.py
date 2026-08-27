# Erwin Lejeune - 2026-08-27
"""Tests for the communication-aware swarming module."""

import numpy as np
import pytest

from uav_sim.comms import (
    ConnectivityController,
    GaussianLink,
    PathLossLink,
    RelayCoverageController,
    algebraic_connectivity,
    connectivity_gradient,
    degree_of_connectivity,
    fiedler_vector,
    hop_counts,
    laplacian,
)


class TestLinkModels:
    @pytest.mark.parametrize(
        "link", [GaussianLink(25.0), PathLossLink(30.0, 2.5, 6.0)], ids=["gaussian", "pathloss"]
    )
    def test_weight_decreases_with_range(self, link):
        d = np.array([1.0, 10.0, 30.0, 90.0])
        w = link.weight(d)
        assert np.all(np.diff(w) < 0.0)
        assert np.all((w >= 0.0) & (w <= 1.0))

    @pytest.mark.parametrize(
        "link", [GaussianLink(25.0), PathLossLink(30.0, 2.5, 6.0)], ids=["gaussian", "pathloss"]
    )
    def test_derivative_matches_finite_difference(self, link):
        """A wrong derivative would still produce plausible-looking motion."""
        d = np.array([5.0, 15.0, 30.0, 60.0])
        h = 1e-7
        numeric = (link.weight(d + h) - link.weight(d - h)) / (2.0 * h)
        np.testing.assert_allclose(link.dweight_ddistance(d), numeric, atol=1e-6)

    def test_weights_matrix_is_symmetric_with_zero_diagonal(self):
        p = np.random.default_rng(0).uniform(0.0, 50.0, (5, 3))
        w = GaussianLink(20.0).weights(p)
        np.testing.assert_allclose(w, w.T)
        np.testing.assert_allclose(np.diag(w), np.zeros(5))

    @pytest.mark.parametrize(
        "factory,match",
        [
            (lambda: GaussianLink(0.0), "sigma must be positive"),
            (lambda: PathLossLink(-1.0), "reference_range must be positive"),
            (lambda: PathLossLink(10.0, exponent=0.0), "exponent must be positive"),
            (lambda: PathLossLink(10.0, softness=0.0), "softness must be positive"),
        ],
    )
    def test_rejects_nonsense(self, factory, match):
        with pytest.raises(ValueError, match=match):
            factory()


class TestGraphMetrics:
    def test_laplacian_rows_sum_to_zero(self):
        w = GaussianLink(20.0).weights(np.random.default_rng(1).uniform(0, 40, (6, 3)))
        np.testing.assert_allclose(laplacian(w).sum(axis=1), np.zeros(6), atol=1e-12)

    def test_lambda2_positive_iff_connected(self):
        link = GaussianLink(10.0)
        close = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
        split = np.array([[0.0, 0, 0], [5.0, 0, 0], [400.0, 0, 0]])
        assert algebraic_connectivity(link.weights(close)) > 1e-3
        assert algebraic_connectivity(link.weights(split)) < 1e-9

    def test_lambda2_falls_as_the_swarm_spreads(self):
        link = GaussianLink(20.0)
        values = [
            algebraic_connectivity(link.weights(np.array([[0.0, 0, 0], [s, 0, 0], [2 * s, 0, 0]])))
            for s in (5.0, 10.0, 20.0, 40.0)
        ]
        assert all(np.diff(values) < 0.0)

    def test_laplacian_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            laplacian(np.zeros((2, 3)))

    def test_fiedler_vector_is_orthogonal_to_ones(self):
        w = GaussianLink(20.0).weights(np.random.default_rng(2).uniform(0, 40, (7, 3)))
        v = fiedler_vector(w)
        assert abs(float(v.sum())) < 1e-9
        assert float(np.linalg.norm(v)) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "link", [GaussianLink(25.0), PathLossLink(30.0, 2.5, 6.0)], ids=["gaussian", "pathloss"]
    )
    def test_connectivity_gradient_matches_finite_difference(self, link):
        """The gradient is the controller; an error here is silent."""
        p = np.random.default_rng(0).uniform(0.0, 60.0, (6, 3))
        analytic = connectivity_gradient(p, link)
        numeric = np.zeros_like(p)
        h = 1e-6
        for i in range(len(p)):
            for k in range(3):
                a, b = p.copy(), p.copy()
                a[i, k] += h
                b[i, k] -= h
                numeric[i, k] = (
                    algebraic_connectivity(link.weights(a))
                    - algebraic_connectivity(link.weights(b))
                ) / (2.0 * h)
        rel = np.max(np.abs(analytic - numeric)) / max(np.max(np.abs(numeric)), 1e-12)
        assert rel < 1e-5

    def test_gradient_is_zero_for_a_single_agent(self):
        np.testing.assert_allclose(
            connectivity_gradient(np.zeros((1, 3)), GaussianLink(20.0)), np.zeros((1, 3))
        )


class TestResilienceAndReachability:
    def test_chain_is_one_connected(self):
        """Every agent in a chain is a single point of failure."""
        p = np.array([[float(i) * 10.0, 0.0, 0.0] for i in range(5)])
        assert degree_of_connectivity(GaussianLink(9.0).weights(p), 0.25) == 1

    def test_disconnected_graph_scores_zero(self):
        p = np.array([[0.0, 0, 0], [10.0, 0, 0], [900.0, 0, 0]])
        assert degree_of_connectivity(GaussianLink(9.0).weights(p), 0.25) == 0

    def test_hop_counts_along_a_chain(self):
        p = np.array([[float(i) * 10.0, 0.0, 0.0] for i in range(4)])
        hops = hop_counts(GaussianLink(9.0).weights(p), 0, 0.25)
        np.testing.assert_allclose(hops, [0.0, 1.0, 2.0, 3.0])

    def test_unreachable_agents_are_infinite(self):
        p = np.array([[0.0, 0, 0], [10.0, 0, 0], [900.0, 0, 0]])
        hops = hop_counts(GaussianLink(9.0).weights(p), 0, 0.25)
        assert np.isinf(hops[2])


class TestConnectivityController:
    def _run(self, keep_connected, steps=1400):
        n, dt = 24, 0.05
        rng = np.random.default_rng(11)
        link = GaussianLink(30.0)
        p = rng.uniform(140.0, 160.0, (n, 3))
        p[:, 2] = 40.0
        goal = np.column_stack([rng.uniform(0, 400, n), rng.uniform(0, 400, n), np.full(n, 40.0)])
        v = np.zeros_like(p)
        ctrl = ConnectivityController(link, lambda_min=0.25, gain=25.0, max_force=12.0)
        for _ in range(steps):
            u = np.clip(0.9 * (goal - p) - 1.8 * v, -4.0, 4.0)
            if keep_connected:
                u = u + ctrl.forces(p)
            v = v + u * dt
            p = p + v * dt
        return algebraic_connectivity(link.weights(p))

    def test_task_alone_fragments_the_network(self):
        assert self._run(keep_connected=False) < 1e-3

    def test_controller_keeps_the_network_alive(self):
        assert self._run(keep_connected=True) > 0.1

    def test_no_force_for_a_lone_agent(self):
        ctrl = ConnectivityController(GaussianLink(20.0))
        np.testing.assert_allclose(ctrl.forces(np.zeros((1, 3))), np.zeros((1, 3)))

    def test_force_is_saturated(self):
        ctrl = ConnectivityController(GaussianLink(20.0), lambda_min=0.5, max_force=3.0)
        # Nearly split, so the barrier term is enormous.
        p = np.array([[0.0, 0, 0], [100.0, 0, 0]])
        assert np.all(np.linalg.norm(ctrl.forces(p), axis=1) <= 3.0 + 1e-9)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"lambda_min": 0.0}, "lambda_min must be positive"),
            ({"gain": -1.0}, "gain must be positive"),
        ],
    )
    def test_rejects_nonsense(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ConnectivityController(GaussianLink(20.0), **kwargs)


class TestRelayCoverage:
    def _run(self, tether, steps=2400):
        n, dt = 18, 0.05
        base = np.array([150.0, 150.0, 40.0])
        link = GaussianLink(34.0)
        ctrl = RelayCoverageController(
            link,
            spread_gain=60.0 if tether > 0 else 200.0,
            anchor_gain=0.5 if tether > 0 else 1.5,
            tether_gain=tether,
            max_force=5.0,
        )
        rng = np.random.default_rng(3)
        p = np.zeros((n, 3))
        p[0] = base
        p[1:] = base + rng.uniform(-12.0, 12.0, (n - 1, 3))
        p[:, 2] = 40.0
        v = np.zeros_like(p)
        for _ in range(steps):
            v = (v + ctrl.forces(p) * dt) * 0.92
            p = p + v * dt
            p[0] = base
            p[:, 2] = 40.0
        hops = hop_counts(link.weights(p), 0, 0.5)
        return p, ctrl, int(np.isfinite(hops).sum())

    def test_base_station_never_moves(self):
        p, _, _ = self._run(tether=25.0, steps=200)
        np.testing.assert_allclose(p[0], [150.0, 150.0, 40.0])

    def test_tethered_fleet_stays_reachable(self):
        _, _, reachable = self._run(tether=25.0)
        assert reachable == 18

    def test_untethered_fleet_loses_the_base(self):
        """Without the tether the swarm spreads itself into uselessness."""
        _, _, reachable = self._run(tether=0.0)
        assert reachable < 5

    def test_coverage_excludes_unreachable_agents(self):
        """Ground watched by an agent that cannot report is not covered."""
        _, ctrl, _ = self._run(tether=0.0, steps=10)
        link = GaussianLink(34.0)
        far = np.array([[150.0, 150.0, 40.0], [1000.0, 1000.0, 40.0]])
        ctrl_local = RelayCoverageController(link, link_threshold=0.5)
        # The distant agent is unreachable, so only the base contributes.
        covered = ctrl_local.coverage_fraction(far, (0.0, 300.0), 25.0, 6.0)
        base_only = ctrl_local.coverage_fraction(far[:1], (0.0, 300.0), 25.0, 6.0)
        assert covered == pytest.approx(base_only)
