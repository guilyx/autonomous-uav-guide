# Erwin Lejeune - 2026-02-15
"""Integration tests for cross-subsystem behaviour.

Unlike unit tests that exercise individual classes, these verify that
multi-component pipelines produce correct end-to-end results:

- Gimbal + Detector: a gimbal pointed at a target actually reports visible.
- BBox tracker loop: detector → tracker → gimbal converges.
- Visual servo loop: servo commands converge the drone toward the target.
- StateManager lifecycle: ARM → TAKEOFF → TRACK → LAND completes.
- Path planning + tracking: planned waypoints are flyable by PID controller.
- Swarm convergence: agents driven by a swarm algorithm converge.
"""

from __future__ import annotations

import numpy as np

from uav_sim.control import FlightMode, StateManager
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.perception.bbox_tracker import (
    SimulatedDetector,
    VisualServoConfig,
    VisualServoController,
)
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import (
    BBoxTracker,
    BBoxTrackerConfig,
    PointTracker,
    project_to_image,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

# ---------------------------------------------------------------------------
# Gimbal + Detector pipeline
# ---------------------------------------------------------------------------


class TestGimbalDetectorIntegration:
    """Verify that the gimbal, detector, and projection pipeline mesh."""

    def test_pointed_gimbal_sees_target(self) -> None:
        """A gimbal aimed at a target must report it as visible."""
        gimbal = Gimbal(max_rate=100.0)
        cam_pos = np.array([10.0, 10.0, 15.0])
        target = np.array([10.0, 10.0, 0.5])

        des_p, des_t = gimbal.look_at(cam_pos, target, 0.0)
        gimbal.reset(pan=des_p, tilt=des_t)

        h_fov, v_fov = 0.8, 0.6
        ndc, visible = project_to_image(target, cam_pos, gimbal, h_fov, v_fov)
        assert visible, "Target should be visible when gimbal is aimed at it"
        assert abs(ndc[0]) < 0.05
        assert abs(ndc[1]) < 0.05

    def test_detector_reports_visible_when_aimed(self) -> None:
        gimbal = Gimbal(max_rate=100.0)
        cam_pos = np.array([15.0, 15.0, 12.0])
        target = np.array([20.0, 15.0, 1.0])

        des_p, des_t = gimbal.look_at(cam_pos, target, 0.0)
        gimbal.reset(pan=des_p, tilt=des_t)

        detector = SimulatedDetector(target_radius=0.5)
        det = detector.detect(target, cam_pos, gimbal, 0.8, 0.6, 0.0)
        assert det.visible
        assert det.size_ratio > 0.0

    def test_point_tracker_converges(self) -> None:
        """PointTracker should slew the gimbal until the target is centred."""
        gimbal = Gimbal(max_rate=3.0)
        gimbal.reset(pan=0.5, tilt=-0.2)
        tracker = PointTracker(gimbal)

        cam_pos = np.array([15.0, 15.0, 15.0])
        target = np.array([20.0, 20.0, 1.0])

        for _ in range(200):
            tracker.step(cam_pos, target, yaw=0.0, dt=0.02)

        ndc, visible = project_to_image(target, cam_pos, gimbal, 0.8, 0.6)
        assert visible, "After convergence, target must be in FOV"
        assert abs(ndc[0]) < 0.15
        assert abs(ndc[1]) < 0.15

    def test_bbox_tracker_centres_target(self) -> None:
        """BBoxTracker in closed loop should drive detection error to zero."""
        gimbal = Gimbal(max_rate=3.0)
        cam_pos = np.array([15.0, 15.0, 15.0])
        target = np.array([18.0, 15.0, 1.0])

        des_p, des_t = gimbal.look_at(cam_pos, target, 0.0)
        gimbal.reset(pan=des_p + 0.1, tilt=des_t + 0.05)

        detector = SimulatedDetector(target_radius=0.5)
        bbox_ctrl = BBoxTracker(gimbal, BBoxTrackerConfig(kp_pan=2.5, kp_tilt=2.5))
        h_fov, v_fov = 0.8, 0.6

        for _ in range(300):
            det = detector.detect(target, cam_pos, gimbal, h_fov, v_fov)
            if det.visible:
                bbox_ctrl.step(det.center_ndc, det.size_ratio, 0.02)
            else:
                dp, dt_ = gimbal.look_at(cam_pos, target, 0.0)
                gimbal.step(dp, dt_, 0.02)

        det_final = detector.detect(target, cam_pos, gimbal, h_fov, v_fov)
        assert det_final.visible, "Target should be visible after tracking"
        assert abs(det_final.center_ndc[0]) < 0.2
        assert abs(det_final.center_ndc[1]) < 0.2


# ---------------------------------------------------------------------------
# Visual servoing convergence
# ---------------------------------------------------------------------------


class TestVisualServoConvergence:
    """Visual servo loop should drive a kinematic drone toward the target."""

    def test_distance_bounded(self) -> None:
        """Drone should not diverge and should stay near the target."""
        gimbal = Gimbal(max_rate=3.0)
        pos = np.array([15.0, 15.0, 10.0])
        target = np.array([20.0, 20.0, 0.5])

        init_p, init_t = gimbal.look_at(pos, target, 0.0)
        gimbal.reset(pan=init_p, tilt=init_t)

        tracker = PointTracker(gimbal)
        detector = SimulatedDetector(target_radius=0.5)
        servo = VisualServoController(
            VisualServoConfig(
                kp_lateral=3.0,
                kp_forward=2.0,
                desired_size_ratio=0.08,
                max_velocity=2.0,
            )
        )

        vel = np.zeros(3)
        h_fov, v_fov = 0.8, 0.6
        initial_dist = float(np.linalg.norm(pos - target))
        max_dist = initial_dist

        for _ in range(500):
            tracker.step(pos, target, 0.0, 0.02)
            det = detector.detect(target, pos, gimbal, h_fov, v_fov)
            vel_cmd = servo.compute(det, 0.0)
            vel = 0.7 * vel + 0.3 * vel_cmd
            pos = pos + vel * 0.02
            pos[2] = np.clip(pos[2], 3.0, 20.0)
            max_dist = max(max_dist, float(np.linalg.norm(pos - target)))

        assert (
            max_dist < initial_dist * 2.0
        ), f"Drone diverged: max distance {max_dist:.1f} > 2× initial {initial_dist:.1f}"

    def test_no_divergence(self) -> None:
        gimbal = Gimbal(max_rate=3.0)
        pos = np.array([15.0, 15.0, 10.0])
        target = np.array([15.0, 15.0, 0.5])

        init_p, init_t = gimbal.look_at(pos, target, 0.0)
        gimbal.reset(pan=init_p, tilt=init_t)
        tracker = PointTracker(gimbal)
        detector = SimulatedDetector(target_radius=0.5)
        servo = VisualServoController(
            VisualServoConfig(kp_lateral=3.0, kp_forward=2.0, max_velocity=2.0)
        )
        vel = np.zeros(3)

        for _ in range(300):
            tracker.step(pos, target, 0.0, 0.02)
            det = detector.detect(target, pos, gimbal, 0.8, 0.6)
            vel_cmd = servo.compute(det, 0.0)
            vel = 0.7 * vel + 0.3 * vel_cmd
            pos = pos + vel * 0.02
            pos[2] = np.clip(pos[2], 3.0, 20.0)

        dist = float(np.linalg.norm(pos[:2] - target[:2]))
        assert dist < 30.0, f"Drone diverged: {dist:.1f} m"


# ---------------------------------------------------------------------------
# StateManager full lifecycle
# ---------------------------------------------------------------------------


class TestStateManagerLifecycle:
    """ARM → TAKEOFF → TRACK → LAND must complete without divergence."""

    def test_full_mission_completes(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.0]))
        sm = StateManager(quad)

        assert sm.arm() is True
        assert sm.mode == FlightMode.ARMED

        sm.run_takeoff(altitude=8.0, dt=0.005, timeout=15.0)
        assert sm.mode == FlightMode.HOVER
        assert abs(quad.position[2] - 8.0) < 2.0

        wps = np.array([[5.0, 0.0, 8.0], [10.0, 0.0, 8.0]])
        sm.track_path(wps, dt=0.005, timeout=30.0)
        assert sm.mode == FlightMode.HOVER

        sm.run_land(dt=0.005, timeout=20.0)
        assert sm.mode == FlightMode.DISARMED
        assert quad.position[2] < 1.0

    def test_states_are_recorded(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.0]))
        sm = StateManager(quad)
        sm.arm()
        sm.run_takeoff(altitude=5.0, dt=0.005, timeout=10.0)
        sm.run_land(dt=0.005, timeout=15.0)
        assert len(sm.states) > 200
        arr = np.array(sm.states)
        assert not np.any(np.isnan(arr)), "No NaNs allowed"

    def test_emergency_from_hover(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 0.0]))
        sm = StateManager(quad)
        sm.arm()
        sm.run_takeoff(altitude=5.0, dt=0.005, timeout=10.0)
        assert sm.emergency() is True
        assert sm.mode == FlightMode.EMERGENCY


# ---------------------------------------------------------------------------
# Planning + tracking end-to-end
# ---------------------------------------------------------------------------


class TestPlanningPlusTracking:
    """Waypoint path + PID + Pure Pursuit → drone reaches the goal."""

    def test_multi_leg_path_reaches_goal(self) -> None:
        quad = Quadrotor()
        start = np.array([5.0, 5.0, 12.0])
        quad.reset(position=start)
        ctrl = CascadedPIDController()
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))

        path = np.array(
            [
                [5, 5, 12],
                [15, 5, 12],
                [15, 20, 12],
                [25, 20, 12.0],
            ]
        )
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(
            quad,
            ctrl,
            path,
            dt=0.005,
            pursuit=pp,
            timeout=60.0,
            states=states,
        )

        arr = np.array(states)
        dist = float(np.linalg.norm(arr[-1, :3] - path[-1]))
        assert dist < 4.0, f"Did not reach goal: {dist:.1f} m away"

    def test_altitude_maintained_on_horizontal_path(self) -> None:
        quad = Quadrotor()
        start = np.array([5.0, 5.0, 12.0])
        quad.reset(position=start)
        ctrl = CascadedPIDController()
        hover_f = quad.hover_wrench()[0] / 4.0
        for m in quad.motors:
            m.reset(m.thrust_to_omega(hover_f))

        path = np.array([[5, 5, 12], [15, 15, 12], [25, 5, 12.0]])
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0)
        states: list[np.ndarray] = []
        fly_path(
            quad,
            ctrl,
            path,
            dt=0.005,
            pursuit=pp,
            timeout=50.0,
            states=states,
        )

        arr = np.array(states)
        mid = len(arr) // 4
        alt_range = arr[mid:, 2].max() - arr[mid:, 2].min()
        assert alt_range < 3.0, f"Altitude range {alt_range:.1f} m"


# ---------------------------------------------------------------------------
# Swarm convergence
# ---------------------------------------------------------------------------


class TestSwarmConvergence:
    """Agents driven by swarm algorithms should converge toward formation."""

    def test_consensus_reduces_spread(self) -> None:
        from uav_sim.swarm.consensus_formation import ConsensusFormation

        n_agents = 4
        rng = np.random.default_rng(42)
        positions = rng.uniform(0, 100, (n_agents, 3))
        velocities = np.zeros((n_agents, 3))
        offsets = np.array(
            [
                [0, 0, 0],
                [5, 0, 0],
                [0, 5, 0],
                [5, 5, 0.0],
            ]
        )
        adjacency = np.ones((n_agents, n_agents)) - np.eye(n_agents)

        ctrl = ConsensusFormation(adjacency=adjacency, offsets=offsets, gain=1.0)
        initial_spread = float(np.std(positions))

        for _ in range(500):
            forces = ctrl.compute_forces(positions)
            velocities += forces * 0.05
            velocities *= 0.95
            positions += velocities * 0.05

        final_spread = float(np.std(positions))
        assert (
            final_spread < initial_spread * 0.5
        ), f"Spread should decrease: {initial_spread:.1f} → {final_spread:.1f}"

    def test_reynolds_agents_dont_diverge(self) -> None:
        from uav_sim.swarm.reynolds_flocking import ReynoldsFlocking

        n_agents = 5
        rng = np.random.default_rng(123)
        positions = rng.uniform(20, 80, (n_agents, 3))
        velocities = rng.uniform(-1, 1, (n_agents, 3))

        flock = ReynoldsFlocking(r_percept=30.0)

        for _ in range(200):
            forces = flock.compute_forces(positions, velocities)
            velocities += forces * 0.05
            speed = np.linalg.norm(velocities, axis=1, keepdims=True)
            velocities = np.where(speed > 3.0, velocities / speed * 3.0, velocities)
            positions += velocities * 0.05

        spread = float(np.std(positions))
        assert spread < 200.0, f"Agents diverged: spread={spread:.1f}"
