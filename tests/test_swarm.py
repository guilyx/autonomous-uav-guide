# Erwin Lejeune - 2026-02-16
"""Tests for swarm algorithms."""

import numpy as np
import pytest

from uav_sim.swarm.consensus_formation import ConsensusFormation
from uav_sim.swarm.coverage import CoverageController
from uav_sim.swarm.leader_follower import LeaderFollower
from uav_sim.swarm.potential_swarm import PotentialSwarm
from uav_sim.swarm.reynolds_flocking import ReynoldsFlocking
from uav_sim.swarm.virtual_structure import VirtualStructure


class TestReynoldsFlocking:
    def test_output_shape(self):
        flock = ReynoldsFlocking()
        pos = np.random.default_rng(42).uniform(0, 10, (5, 3))
        vel = np.random.default_rng(42).uniform(-1, 1, (5, 3))
        forces = flock.compute_forces(pos, vel)
        assert forces.shape == (5, 3)

    def test_separation_pushes_apart(self):
        flock = ReynoldsFlocking(r_percept=10, r_sep=5, w_sep=5.0, w_ali=0, w_coh=0)
        pos = np.array([[0, 0, 0], [0.5, 0, 0.0]])
        vel = np.zeros((2, 3))
        forces = flock.compute_forces(pos, vel)
        # Agent 0 should be pushed in -x direction (away from agent 1).
        assert forces[0, 0] < 0
        # Agent 1 should be pushed in +x direction.
        assert forces[1, 0] > 0

    def test_cohesion_pulls_together(self):
        flock = ReynoldsFlocking(r_percept=20, r_sep=0.1, w_sep=0, w_ali=0, w_coh=5.0)
        pos = np.array([[0, 0, 0], [10, 0, 0.0]])
        vel = np.zeros((2, 3))
        forces = flock.compute_forces(pos, vel)
        # Agent 0 pulled towards agent 1 (+x).
        assert forces[0, 0] > 0

    def test_no_neighbours_gives_zero_force(self):
        flock = ReynoldsFlocking(r_percept=1.0)
        pos = np.array([[0, 0, 0], [100, 0, 0.0]])
        vel = np.zeros((2, 3))
        forces = flock.compute_forces(pos, vel)
        np.testing.assert_array_equal(forces, 0.0)


class TestConsensusFormation:
    def test_converges_to_formation(self):
        N = 3
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]])
        offsets = np.array([[0, 0, 0], [2, 0, 0], [1, 2, 0.0]])
        ctrl = ConsensusFormation(adj, offsets, gain=1.0)

        pos = np.random.default_rng(42).uniform(-5, 5, (N, 3))
        for _ in range(500):
            f = ctrl.compute_forces(pos)
            pos += 0.01 * f

        # Agents should converge towards offsets + common translation.
        diffs = pos - pos[0] - (offsets - offsets[0])
        assert np.max(np.abs(diffs)) < 0.5

    def test_output_shape(self):
        adj = np.ones((4, 4)) - np.eye(4)
        offsets = np.zeros((4, 3))
        ctrl = ConsensusFormation(adj, offsets)
        forces = ctrl.compute_forces(np.random.default_rng(42).uniform(0, 1, (4, 3)))
        assert forces.shape == (4, 3)


class TestVirtualStructure:
    def test_desired_positions_at_origin(self):
        offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0.0]])
        vs = VirtualStructure(offsets)
        des = vs.desired_positions(np.array([0.0, 0.0, 0.0]), body_yaw=0.0)
        np.testing.assert_allclose(des, offsets, atol=1e-10)

    def test_forces_drive_to_formation(self):
        offsets = np.array([[1, 0, 0], [-1, 0, 0.0]])
        vs = VirtualStructure(offsets, kp=5.0, kd=2.0)
        pos = np.zeros((2, 3))
        vel = np.zeros((2, 3))
        forces = vs.compute_forces(pos, vel, body_pos=np.array([0.0, 0.0, 0.0]))
        # Agent 0 should be pushed towards [1,0,0].
        assert forces[0, 0] > 0
        # Agent 1 should be pushed towards [-1,0,0].
        assert forces[1, 0] < 0


class TestLeaderFollower:
    def test_output_shape(self):
        offsets = np.array([[2, 0, 0], [-2, 0, 0], [0, 2, 0.0]])
        lf = LeaderFollower(offsets)
        forces = lf.compute_forces(np.zeros(3), np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)))
        assert forces.shape == (3, 3)

    def test_follower_attracted_to_offset(self):
        offsets = np.array([[2, 0, 0.0]])
        lf = LeaderFollower(offsets, kp=5.0)
        leader_pos = np.array([0, 0, 0.0])
        follower_pos = np.array([[0, 0, 0.0]])
        forces = lf.compute_forces(leader_pos, np.zeros(3), follower_pos, np.zeros((1, 3)))
        # Follower at origin, desired at [2,0,0] → force in +x.
        assert forces[0, 0] > 0


class TestPotentialSwarm:
    def test_equilibrium_distance(self):
        a, b, d_des = 4, 2, 2.0
        # True LJ equilibrium: r_eq = d_des * (a/b)^(1/(a-b))
        r_eq = d_des * (a / b) ** (1 / (a - b))
        ps = PotentialSwarm(d_des=d_des, epsilon=5.0, a=a, b=b, goal_gain=0.0)
        pos = np.array([[0, 0, 0], [r_eq, 0, 0.0]])
        forces = ps.compute_forces(pos)
        np.testing.assert_allclose(forces, 0.0, atol=1e-10)

    def test_too_close_repels(self):
        ps = PotentialSwarm(d_des=3.0, epsilon=5.0, goal_gain=0.0)
        pos = np.array([[0, 0, 0], [1.0, 0, 0.0]])  # closer than desired
        forces = ps.compute_forces(pos)
        # Agent 0 pushed in -x, agent 1 in +x.
        assert forces[0, 0] < 0
        assert forces[1, 0] > 0


class TestCoverage:
    def test_centroids_shape(self):
        bounds = np.array([[0, 0], [10, 10.0]])
        cc = CoverageController(bounds, resolution=1.0)
        pos = np.array([[2, 2], [8, 8.0]])
        centroids = cc.compute_centroids(pos)
        assert centroids.shape == (2, 2)

    def test_forces_drive_towards_centroid(self):
        bounds = np.array([[0, 0], [10, 10.0]])
        cc = CoverageController(bounds, resolution=0.5, gain=1.0)
        pos = np.array([[0.5, 0.5], [9.5, 9.5]])
        forces = cc.compute_forces(pos)
        # Agent 0 in corner should be pushed towards centre of its cell.
        assert forces[0, 0] > 0
        assert forces[0, 1] > 0

    def test_single_agent_returns_own_position(self):
        bounds = np.array([[0, 0], [10, 10.0]])
        cc = CoverageController(bounds, resolution=1.0)
        pos = np.array([[5, 5.0]])
        centroids = cc.compute_centroids(pos)
        # Single agent: early return gives back position unchanged.
        np.testing.assert_allclose(centroids[0], pos[0])

    def test_two_agents_centroids_separate(self):
        bounds = np.array([[0, 0], [10, 10.0]])
        cc = CoverageController(bounds, resolution=0.5)
        pos = np.array([[2.5, 5.0], [7.5, 5.0]])
        centroids = cc.compute_centroids(pos)
        # Each agent should own roughly half the workspace; centroids should
        # be on opposite sides of x = 5.
        assert centroids[0, 0] < 5.0
        assert centroids[1, 0] > 5.0


class TestCoverageMovingRegion:
    """A fixed-size region that travels, rather than a static workspace."""

    def _ctrl(self):
        return CoverageController(
            bounds=np.array([[0.0, 0.0], [20.0, 20.0]]), resolution=1.0, gain=0.5
        )

    def test_recenter_moves_the_region(self):
        cc = self._ctrl()
        cc.recenter(np.array([70.0, 40.0]))
        np.testing.assert_allclose(cc.region_center, [70.0, 40.0])
        np.testing.assert_allclose(cc.bounds, [[60.0, 30.0], [80.0, 50.0]])

    def test_recenter_preserves_the_grid(self):
        """Re-meshing would change the point count and jolt the cost."""
        cc = self._ctrl()
        n_before = len(cc.grid)
        cc.recenter(np.array([70.3, 40.7]))
        assert len(cc.grid) == n_before

    def test_recenter_accepts_a_3d_guide_point(self):
        cc = self._ctrl()
        cc.recenter(np.array([70.0, 40.0, 50.0]))
        np.testing.assert_allclose(cc.region_center, [70.0, 40.0])

    def test_cost_is_translation_invariant(self):
        cc = self._ctrl()
        pos = np.array([[5.0, 5.0], [15.0, 15.0]])
        before = cc.coverage_cost(pos)
        cc.recenter(np.array([70.0, 40.0]))
        after = cc.coverage_cost(pos + (cc.region_center - np.array([10.0, 10.0])))
        assert after == pytest.approx(before)

    def test_recenter_rejects_a_scalar_centre(self):
        with pytest.raises(ValueError, match="at least 2 components"):
            self._ctrl().recenter(np.array([1.0]))

    def test_pure_lloyd_strands_an_agent_outside_the_region(self):
        """An empty Voronoi cell yields zero force — the documented flaw."""
        cc = self._ctrl()
        pos = np.array([[10.0, 10.0], [95.0, 95.0]])
        assert cc.empty_cells(pos).tolist() == [False, True]
        forces = cc.compute_forces(pos, recall_outside=False)
        np.testing.assert_allclose(forces[1], [0.0, 0.0])

    def test_recall_steers_a_stranded_agent_back(self):
        cc = self._ctrl()
        pos = np.array([[10.0, 10.0], [95.0, 95.0]])
        forces = cc.compute_forces(pos, recall_outside=True)
        assert np.linalg.norm(forces[1]) > 0.0
        # and it must point towards the region, not merely be non-zero
        assert float(np.dot(forces[1], cc.region_center - pos[1])) > 0.0

    def test_recall_does_not_disturb_agents_inside(self):
        cc = self._ctrl()
        pos = np.array([[6.0, 6.0], [14.0, 14.0]])
        assert not cc.empty_cells(pos).any()
        np.testing.assert_allclose(
            cc.compute_forces(pos, recall_outside=True),
            cc.compute_forces(pos, recall_outside=False),
        )
