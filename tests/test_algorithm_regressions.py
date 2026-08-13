# Erwin Lejeune - 2026-02-22
"""Regression tests for defects found while auditing every simulation.

Each test here pins down one specific way an algorithm was silently wrong
— not wrong enough to crash or fail a shape check, but wrong enough that
the rendered demo showed the wrong behaviour.
"""

import numpy as np
import pytest

from uav_sim.estimation.particle_filter import ParticleFilter
from uav_sim.estimation.process_noise import (
    constant_acceleration_input_q,
    constant_velocity_q,
)
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.path_tracking.geometric_controller import (
    GeometricController,
    GeometricControllerConfig,
)
from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.path_tracking.mpc_controller import MPCController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.perception.bbox_tracker import (
    Detection,
    VisualServoConfig,
    VisualServoController,
)
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import BBoxTracker, project_to_image
from uav_sim.sensors.imu import IMU
from uav_sim.simulations.standards import figure_8_reference
from uav_sim.swarm.coverage import CoverageController
from uav_sim.swarm.potential_swarm import PotentialSwarm
from uav_sim.swarm.reynolds_flocking import ReynoldsFlocking
from uav_sim.swarm.virtual_structure import VirtualStructure
from uav_sim.trajectory_tracking.feedback_linearisation import FeedbackLinearisationTracker
from uav_sim.trajectory_tracking.mppi import MPPITracker
from uav_sim.trajectory_tracking.nmpc import NMPCTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class TestCameraFrameHandedness:
    """The gimbal's camera frame was left-handed, mirroring every image."""

    def test_rotation_matrix_is_a_rotation(self):
        g = Gimbal()
        for pan, tilt in [(0.0, 0.0), (0.7, -0.4), (-2.0, 0.3)]:
            g.reset(pan=pan, tilt=tilt)
            R = g.rotation_matrix(yaw=0.3)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)

    def test_camera_axes_are_right_down_forward(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=0.0)
        R = g.rotation_matrix(yaw=0.0)
        # Looking along world +x: right is -y, down is -z, forward is +x.
        np.testing.assert_allclose(R[:, 0], [0.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(R[:, 1], [0.0, 0.0, -1.0], atol=1e-9)
        np.testing.assert_allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-9)

    def test_target_to_the_right_projects_to_positive_x(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=0.0)
        cam = np.array([0.0, 0.0, 10.0])
        # +x is forward, so "to the right" of the camera is -y.
        right_target = np.array([10.0, -2.0, 10.0])
        ndc, visible = project_to_image(right_target, cam, g, h_fov=1.0, v_fov=0.8)
        assert visible
        assert ndc[0] > 0.0

    def test_target_below_projects_to_positive_y(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=0.0)
        cam = np.array([0.0, 0.0, 10.0])
        ndc, visible = project_to_image(
            np.array([10.0, 0.0, 8.0]), cam, g, h_fov=1.0, v_fov=0.8
        )
        assert visible
        assert ndc[1] > 0.0


class TestGimbalPanSign:
    def test_pan_turns_towards_a_target_on_the_right(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=0.0)
        bt = BBoxTracker(g)
        # Target right of centre → camera must swing right → pan decreases.
        for _ in range(5):
            bt.step(np.array([0.6, 0.0]), 0.2, dt=0.05)
        assert g.pan < 0.0

    def test_pan_turns_towards_a_target_on_the_left(self):
        g = Gimbal()
        g.reset(pan=0.0, tilt=0.0)
        bt = BBoxTracker(g)
        for _ in range(5):
            bt.step(np.array([-0.6, 0.0]), 0.2, dt=0.05)
        assert g.pan > 0.0


class TestVisualServoSigns:
    def test_target_right_of_centre_commands_motion_to_the_right(self):
        ctrl = VisualServoController(VisualServoConfig(kp_lateral=1.0, kp_forward=0.0))
        det = Detection(center_ndc=np.array([0.5, 0.0]), size_ratio=0.25, visible=True)
        vel = ctrl.compute(det, yaw=0.0)
        # Body +y is left (FLU), so chasing right means negative world y.
        assert vel[1] < 0.0

    def test_target_below_centre_commands_descent(self):
        ctrl = VisualServoController(VisualServoConfig(kp_lateral=1.0, kp_forward=0.0))
        det = Detection(center_ndc=np.array([0.0, 0.5]), size_ratio=0.25, visible=True)
        vel = ctrl.compute(det, yaw=0.0)
        assert vel[2] < 0.0

    def test_gimbal_mode_uses_pan_not_image_centre(self):
        cfg = VisualServoConfig(kp_forward=0.0, kp_pan=1.0, kp_tilt=0.0, desired_tilt=0.0)
        ctrl = VisualServoController(cfg)
        # Bbox perfectly centred — a gimbal keeps it there — but pan says
        # the target has drifted to the left of the nose.
        det = Detection(center_ndc=np.zeros(2), size_ratio=0.25, visible=True)
        vel = ctrl.compute_from_gimbal(det, yaw=0.0, pan=0.5, tilt=0.0)
        assert vel[1] > 0.0
        assert np.linalg.norm(ctrl.compute(det, yaw=0.0)) == pytest.approx(0.0, abs=1e-9)


class TestProcessNoiseScaling:
    """A dt-independent Q makes a filter echo its measurements."""

    def test_constant_velocity_q_scales_with_dt(self):
        q_fine = constant_velocity_q(0.005, psd=1.0)
        q_coarse = constant_velocity_q(0.05, psd=1.0)
        # Velocity block grows linearly in dt for continuous white noise.
        assert q_coarse[3, 3] == pytest.approx(10.0 * q_fine[3, 3], rel=1e-9)

    def test_accumulated_covariance_is_step_size_independent(self):
        """Ten fine steps must accumulate what one coarse step does."""
        total_fine = 10 * constant_velocity_q(0.005, psd=2.0)[3, 3]
        one_coarse = constant_velocity_q(0.05, psd=2.0)[3, 3]
        assert total_fine == pytest.approx(one_coarse, rel=1e-9)

    def test_q_is_positive_semidefinite(self):
        for q in (
            constant_velocity_q(0.01, psd=1.5),
            constant_acceleration_input_q(0.01, sigma_a=0.1, sigma_bias=0.05),
        ):
            eigs = np.linalg.eigvalsh(q)
            assert eigs.min() > -1e-12

    def test_bias_term_only_adds_velocity_noise(self):
        without = constant_acceleration_input_q(0.01, sigma_a=0.1)
        with_bias = constant_acceleration_input_q(0.01, sigma_a=0.1, sigma_bias=0.05)
        assert with_bias[3, 3] > without[3, 3]
        assert with_bias[0, 0] == pytest.approx(without[0, 0])


class TestIMUSpecificForce:
    def test_level_hover_reads_gravity_on_body_z(self):
        imu = IMU(accel_noise_std=0.0, accel_bias_std=0.0, gyro_bias_std=0.0, seed=0)
        state = np.zeros(12)
        imu.sense(state)  # prime the finite difference
        m = imu.sense(state)
        np.testing.assert_allclose(m[:3], [0.0, 0.0, 9.81], atol=1e-6)

    def test_bias_is_constant_across_calls(self):
        imu = IMU(accel_noise_std=0.0, gyro_noise_std=0.0, seed=3)
        first = imu.accel_bias
        imu.sense(np.zeros(12))
        np.testing.assert_allclose(first, imu.accel_bias)


class TestParticleFilterDeterminism:
    def _build(self, seed):
        pf = ParticleFilter(
            state_dim=2,
            num_particles=64,
            f=lambda x, u, dt: x,
            likelihood=lambda z, x: float(np.exp(-np.sum((z - x) ** 2))),
            process_noise_std=0.1,
            seed=seed,
        )
        pf.reset(np.zeros(2), spread=0.5)
        return pf

    def test_same_seed_gives_same_estimate(self):
        results = []
        for _ in range(2):
            pf = self._build(11)
            for _ in range(10):
                pf.predict(np.zeros(1), 0.1)
                pf.update(np.array([0.2, -0.1]))
            results.append(pf.estimate.copy())
        np.testing.assert_allclose(results[0], results[1])

    def test_effective_sample_size_starts_at_n(self):
        pf = self._build(5)
        assert pf.effective_sample_size == pytest.approx(pf.N)


class TestCoverageBounds:
    def test_transposed_bounds_are_rejected(self):
        # [[x_min, x_max], [y_min, y_max]] silently produced an empty grid
        # and therefore zero coverage force for every agent.
        with pytest.raises(ValueError):
            CoverageController(np.array([[0.0, 100.0], [0.0, 100.0]]), resolution=2.0)

    def test_lloyd_descends_the_coverage_cost(self):
        ctrl = CoverageController(np.array([[0.0, 0.0], [40.0, 40.0]]), resolution=1.0, gain=0.5)
        pos = np.array([[5.0, 5.0], [7.0, 6.0], [6.0, 8.0], [8.0, 8.0]])
        start = ctrl.coverage_cost(pos)
        vel = np.zeros_like(pos)
        for _ in range(200):
            vel = vel * 0.85 + ctrl.compute_forces(pos) * 0.1
            pos = np.clip(pos + vel * 0.1, 1.0, 39.0)
        assert ctrl.coverage_cost(pos) < 0.5 * start

    def test_forces_are_non_zero_for_clustered_agents(self):
        ctrl = CoverageController(np.array([[0.0, 0.0], [20.0, 20.0]]), resolution=1.0)
        forces = ctrl.compute_forces(np.array([[2.0, 2.0], [3.0, 3.0]]))
        assert np.linalg.norm(forces) > 1e-6


class TestFlockMigration:
    def test_flock_keeps_moving_with_a_migratory_urge(self):
        flock = ReynoldsFlocking(
            r_percept=30.0, r_sep=6.0, w_sep=2.0, w_ali=1.2, w_coh=1.0, w_mig=1.5
        )
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 20, (8, 3))
        vel = np.zeros((8, 3))
        migration = np.array([3.0, 0.0, 0.0])
        for _ in range(300):
            f = flock.compute_forces(pos, vel, migration_velocity=migration)
            vel = vel * 0.92 + f * 0.1
            pos = pos + vel * 0.1
        assert np.linalg.norm(vel, axis=1).mean() > 1.0

    def test_flock_stalls_without_one(self):
        """The behaviour that made the old demo look frozen."""
        flock = ReynoldsFlocking(r_percept=30.0, r_sep=6.0, w_sep=2.0, w_ali=1.2, w_coh=1.0)
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 20, (8, 3))
        vel = np.zeros((8, 3))
        for _ in range(400):
            vel = vel * 0.92 + flock.compute_forces(pos, vel) * 0.1
            pos = pos + vel * 0.1
        assert np.linalg.norm(vel, axis=1).mean() < 0.5


class TestVirtualStructureFeedforward:
    def _run(self, use_ff):
        offsets = np.array([[6.0, 0, 0], [-6.0, 0, 0], [0, 6.0, 0], [0, -6.0, 0]])
        ctrl = VirtualStructure(offsets, kp=2.0, kd=1.5)
        pos = np.tile(np.array([0.0, 0.0, 10.0]), (4, 1)) + offsets
        vel = np.zeros((4, 3))
        body_vel = np.array([2.0, 0.0, 0.0])
        errors = []
        for step in range(400):
            body_pos = np.array([2.0 * step * 0.05, 0.0, 10.0])
            forces = (
                ctrl.compute_forces(pos, vel, body_pos, body_vel=body_vel)
                if use_ff
                else ctrl.compute_forces(pos, vel, body_pos)
            )
            vel = vel + forces * 0.05
            pos = pos + vel * 0.05
            errors.append(
                np.linalg.norm(pos - ctrl.desired_positions(body_pos), axis=1).mean()
            )
        return errors[-1]

    def test_velocity_feedforward_removes_the_standing_lag(self):
        with_ff = self._run(use_ff=True)
        without_ff = self._run(use_ff=False)
        assert with_ff < 0.15 * without_ff
        # What is left is the half-step offset of the explicit integrator
        # (v·dt), not controller lag.
        assert with_ff <= 1.5 * 2.0 * 0.05


class TestPotentialSwarmSaturation:
    def test_goal_attraction_saturates(self):
        ctrl = PotentialSwarm(goal_gain=1.0, goal_saturation=5.0)
        near = ctrl.compute_forces(np.array([[0.0, 0.0, 0.0]]), goal=np.array([3.0, 0.0, 0.0]))
        far = ctrl.compute_forces(np.array([[0.0, 0.0, 0.0]]), goal=np.array([300.0, 0.0, 0.0]))
        assert near[0, 0] == pytest.approx(3.0)
        assert far[0, 0] == pytest.approx(5.0)

    def test_obstacle_repulsion_is_bounded_on_contact(self):
        ctrl = PotentialSwarm(obs_gain=50.0, obs_range=3.0, max_force=20.0)
        # Agent inside the obstacle: the unclamped 1/d² term went infinite.
        forces = ctrl.compute_forces(
            np.array([[1.0, 0.0, 0.0]]), obstacles=[(np.array([0.0, 0.0, 0.0]), 2.0)]
        )
        assert np.all(np.isfinite(forces))
        assert np.linalg.norm(forces) <= 20.0 + 1e-9


class TestGeometricFeedforward:
    def _track(self, feedforward):
        quad = Quadrotor()
        pos0, vel0, _ = figure_8_reference(0.0)
        quad.reset(position=pos0.copy(), velocity=vel0.copy())
        init_hover(quad)
        ctrl = GeometricController()
        dt = 0.01
        errors = []
        for i in range(2000):
            t = i * dt
            ref_p, ref_v, ref_a = figure_8_reference(t)
            wrench = ctrl.compute(
                quad.state,
                ref_p,
                target_vel=ref_v,
                target_acc=ref_a if feedforward else None,
                dt=dt if feedforward else 0.0,
            )
            quad.step(wrench, dt)
            if t > 5.0:
                errors.append(float(np.linalg.norm(quad.state[:3] - ref_p)))
        return float(np.mean(errors))

    def test_acceleration_feedforward_shrinks_tracking_error(self):
        assert self._track(feedforward=True) < 0.3 * self._track(feedforward=False)

    def test_tracks_the_figure_eight_amplitude(self):
        quad = Quadrotor()
        pos0, vel0, _ = figure_8_reference(0.0)
        quad.reset(position=pos0.copy(), velocity=vel0.copy())
        init_hover(quad)
        ctrl = GeometricController()
        dt, n = 0.01, 3000
        flown = np.zeros((n, 3))
        for i in range(n):
            ref_p, ref_v, ref_a = figure_8_reference(i * dt)
            flown[i] = quad.state[:3]
            quad.step(ctrl.compute(quad.state, ref_p, ref_v, ref_a, dt=dt), dt)
        # The reference spans 12 m in x and 8 m in y; the old controller
        # overshot the fast axis by ~17%.
        assert flown[:, 0].max() - flown[:, 0].min() == pytest.approx(12.0, abs=0.5)
        assert flown[:, 1].max() - flown[:, 1].min() == pytest.approx(8.0, abs=0.4)

    def test_attitude_gains_scale_with_inertia(self):
        light = GeometricControllerConfig(inertia=np.diag([0.001, 0.001, 0.002]))
        heavy = GeometricControllerConfig(inertia=np.diag([0.01, 0.01, 0.02]))
        assert heavy.kR > 5.0 * light.kR

    def test_attitude_loop_is_faster_than_position_loop(self):
        cfg = GeometricControllerConfig()
        assert cfg.attitude_bandwidth >= 5.0 * cfg.position_bandwidth


class TestTrackerFeedforward:
    """Every figure-8 tracker was fighting the reference acceleration."""

    @staticmethod
    def _fly(step_fn, dt, duration=30.0):
        quad = Quadrotor()
        pos0, vel0, _ = figure_8_reference(0.0)
        quad.reset(position=pos0.copy(), velocity=vel0.copy())
        init_hover(quad)
        n = int(duration / dt)
        flown = np.zeros((n, 3))
        errors = []
        for i in range(n):
            t = i * dt
            ref = figure_8_reference(t)
            flown[i] = quad.state[:3]
            quad.step(step_fn(quad.state, ref), dt)
            if t > 5.0:
                errors.append(float(np.linalg.norm(quad.state[:3] - ref[0])))
        return float(np.mean(errors)), flown

    def test_lqr_feedforward_shrinks_error(self):
        quad = Quadrotor()
        lqr = LQRController(
            mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
        )

        def make(ff):
            def step(state, ref):
                target = np.zeros(12)
                target[:3] = ref[0]
                target[6:9] = ref[1]
                return lqr.compute(state, target, feedforward_acc=ref[2] if ff else None)

            return step

        with_ff, flown = self._fly(make(True), 0.01)
        without_ff, _ = self._fly(make(False), 0.01)
        assert with_ff < 0.3 * without_ff
        assert flown[:, 1].max() - flown[:, 1].min() == pytest.approx(8.0, abs=0.2)

    def test_feedback_linearisation_uses_the_flat_feedforward(self):
        quad = Quadrotor()
        tracker = FeedbackLinearisationTracker(
            mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
        )

        def make(ff):
            def step(state, ref):
                return tracker.compute(state, ref[0], ref[1], ref[2] if ff else np.zeros(3))

            return step

        assert self._fly(make(True), 0.005)[0] < 0.3 * self._fly(make(False), 0.005)[0]

    def test_mpc_preview_shrinks_error(self):
        quad = Quadrotor()
        horizon, ctrl_dt = 8, 0.05
        mpc = MPCController(
            horizon=horizon,
            dt=ctrl_dt,
            mass=quad.params.mass,
            gravity=quad.params.gravity,
            inertia=quad.params.inertia,
        )

        def run(preview):
            mpc.reset()
            pos0, vel0, _ = figure_8_reference(0.0)
            quad.reset(position=pos0.copy(), velocity=vel0.copy())
            init_hover(quad)
            dt, wrench, since = 0.01, quad.hover_wrench(), 0.0
            errors = []
            for i in range(int(20.0 / dt)):
                t = i * dt
                since += dt
                if since >= ctrl_dt - 1e-9:
                    if preview:
                        hor = [figure_8_reference(t + k * ctrl_dt) for k in range(horizon + 1)]
                        wrench = mpc.compute(
                            quad.state,
                            np.array([h[0] for h in hor]),
                            target_vel=np.array([h[1] for h in hor]),
                        )
                    else:
                        ref = figure_8_reference(t)
                        wrench = mpc.compute(quad.state, ref[0], target_vel=ref[1])
                    since = 0.0
                quad.step(wrench, dt)
                if t > 5.0:
                    errors.append(float(np.linalg.norm(quad.state[:3] - figure_8_reference(t)[0])))
            return float(np.mean(errors))

        assert run(preview=True) < 0.3 * run(preview=False)


class TestMPPIReference:
    def test_cost_receives_horizon_step(self):
        seen = []

        def cost(x, u, ref, k):
            seen.append(k)
            return float(np.sum(x[:2] ** 2))

        mppi = MPPITracker(
            state_dim=4,
            control_dim=2,
            horizon=5,
            num_samples=3,
            dynamics=lambda x, u, dt: x,
            cost_fn=cost,
            dt=0.1,
        )
        mppi.compute(np.zeros(4), reference=None, seed=1)
        assert sorted(set(seen)) == [0, 1, 2, 3, 4]

    def test_three_argument_cost_still_works(self):
        mppi = MPPITracker(
            state_dim=4,
            control_dim=2,
            horizon=5,
            num_samples=8,
            dynamics=lambda x, u, dt: x,
            cost_fn=lambda x, u, ref: float(np.sum(u**2)),
            dt=0.1,
        )
        assert mppi.compute(np.zeros(4), reference=None, seed=1).shape == (2,)

    def test_nominal_rollout_has_horizon_length(self):
        def dyn(x, u, dt):
            return np.array(
                [x[0] + x[2] * dt, x[1] + x[3] * dt, x[2] + u[0] * dt, x[3] + u[1] * dt]
            )

        mppi = MPPITracker(
            state_dim=4,
            control_dim=2,
            horizon=7,
            num_samples=16,
            dynamics=dyn,
            cost_fn=lambda x, u, ref, k: float(np.sum(x[:2] ** 2)),
            dt=0.1,
        )
        mppi.compute(np.array([1.0, 1.0, 0.0, 0.0]), reference=None, seed=2)
        assert mppi.nominal_rollout(np.array([1.0, 1.0, 0.0, 0.0])).shape == (7, 4)


class TestNMPCTracking:
    def test_tracks_a_moving_reference(self):
        quad = Quadrotor()
        pos0, vel0, _ = figure_8_reference(0.0)
        quad.reset(position=pos0.copy(), velocity=vel0.copy())
        init_hover(quad)
        nmpc = NMPCTracker(
            horizon=20,
            dt=0.05,
            n_blocks=4,
            mass=quad.params.mass,
            gravity=quad.params.gravity,
            inertia=quad.params.inertia,
        )
        dt_sim, dt_ctrl = 0.01, 0.05
        errors = []
        for ci in range(int(12.0 / dt_ctrl)):
            t = ci * dt_ctrl
            horizon = [figure_8_reference(t + (k + 1) * 0.05) for k in range(20)]
            wrench = nmpc.compute(
                quad.state,
                np.array([h[0] for h in horizon]),
                ref_vel=np.array([h[1] for h in horizon]),
            )
            for _ in range(int(dt_ctrl / dt_sim)):
                quad.step(wrench, dt_sim)
            if t > 4.0:
                errors.append(float(np.linalg.norm(quad.state[:3] - figure_8_reference(t)[0])))
        assert np.mean(errors) < 0.25

    def test_single_point_reference_is_broadcast(self):
        nmpc = NMPCTracker(horizon=6, dt=0.05, n_blocks=2)
        ref = nmpc._build_reference(np.array([1.0, 2.0, 3.0]), None)
        assert ref.shape == (6, 9)
        np.testing.assert_allclose(ref[:, :3], np.tile([1.0, 2.0, 3.0], (6, 1)))

    def test_returns_a_finite_wrench(self):
        nmpc = NMPCTracker(horizon=10, dt=0.05, n_blocks=3)
        wrench = nmpc.compute(np.zeros(12), np.array([2.0, 0.0, 5.0]))
        assert wrench.shape == (4,)
        assert np.all(np.isfinite(wrench))
        assert wrench[0] >= 0.0


class TestPurePursuitGoalThreshold:
    def test_goal_threshold_defaults_to_waypoint_threshold(self):
        pp = PurePursuit3D(waypoint_threshold=1.5)
        assert pp.goal_threshold == pytest.approx(1.5)

    def test_tighter_goal_threshold_keeps_tracking(self):
        path = np.array([[0.0, 0, 0], [10.0, 0, 0]])
        pp = PurePursuit3D(waypoint_threshold=1.5, goal_threshold=0.5)
        # 1.2 m out: inside the waypoint threshold but not yet at the goal.
        assert not pp.is_path_complete(np.array([8.8, 0.0, 0.0]), path)
        assert pp.is_path_complete(np.array([9.7, 0.0, 0.0]), path)


class TestPurePursuitProgress:
    @staticmethod
    def _lawnmower(lanes=6, length=20.0, spacing=1.0, step=1.0):
        """Boustrophedon path whose adjacent lanes pass within `spacing`."""
        pts = []
        for lane in range(lanes):
            y = lane * spacing
            xs = np.arange(0.0, length + step, step)
            if lane % 2:
                xs = xs[::-1]
            pts.extend([[x, y, 10.0] for x in xs])
        return np.array(pts)

    def test_does_not_hop_between_adjacent_lanes(self):
        """An index-based search window skipped most of a coverage path."""
        path = self._lawnmower()
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.0)
        # Sitting on lane 0, the nearest point on lane 1 is 1 m away but
        # a whole lane further along the path.
        pp.compute_target(path[5].copy(), path)
        assert pp.current_index <= 8

    def test_escapes_a_self_intersecting_loop(self):
        """A path that comes back on itself used to trap the tracker."""
        theta = np.linspace(0.0, 4.0 * np.pi, 120)
        path = np.column_stack(
            [5.0 * np.cos(theta), 5.0 * np.sin(theta), np.linspace(10.0, 10.0, 120)]
        )
        pp = PurePursuit3D(lookahead=2.0, waypoint_threshold=1.0)
        # Walk the vehicle along the path; the index must reach the end.
        for p in path[::2]:
            pp.compute_target(p.copy(), path)
        assert pp.current_index >= len(path) - 3

    def test_index_never_moves_backwards(self):
        path = self._lawnmower()
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.0)
        last = 0
        for p in path:
            pp.compute_target(p.copy(), path)
            assert pp.current_index >= last
            last = pp.current_index
