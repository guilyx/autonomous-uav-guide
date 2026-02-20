# Erwin Lejeune - 2026-02-19
"""Integration tests for end-to-end flight operations.

Verifies that the drone actually reaches goals, maintains altitude,
doesn't diverge, and keeps a reasonable velocity — failures that unit
tests on individual controllers would miss.
"""

from __future__ import annotations

import numpy as np

from uav_sim.path_tracking.flight_ops import fly_mission, fly_path, loiter, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

_GOAL_TOL = 2.5  # metres — acceptable distance to goal at end
_MAX_VEL = 4.0  # m/s — drone should never exceed this in normal ops
_MAX_TILT_DEG = 25.0  # degrees — tilt limit for smooth flight


def _setup_hovering_quad(pos: np.ndarray) -> tuple[Quadrotor, CascadedPIDController]:
    """Create a quadrotor at *pos* with motors pre-spun for hover."""
    quad = Quadrotor()
    quad.reset(position=pos)
    ctrl = CascadedPIDController()
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))
    return quad, ctrl


class TestTakeoff:
    def test_reaches_target_altitude(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([5.0, 5.0, 0.0]))
        ctrl = CascadedPIDController()
        states: list[np.ndarray] = []
        takeoff(quad, ctrl, target_alt=10.0, dt=0.005, duration=5.0, states=states)
        final_z = states[-1][2]
        assert abs(final_z - 10.0) < 1.5, f"Takeoff altitude {final_z:.2f} != 10.0 ± 1.5"

    def test_stays_near_xy_position(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([10.0, 10.0, 0.0]))
        ctrl = CascadedPIDController()
        states: list[np.ndarray] = []
        takeoff(quad, ctrl, target_alt=15.0, dt=0.005, duration=5.0, states=states)
        arr = np.array(states)
        xy_drift = np.max(np.linalg.norm(arr[:, :2] - np.array([10.0, 10.0]), axis=1))
        assert xy_drift < 1.0, f"XY drift during takeoff: {xy_drift:.2f} > 1.0 m"

    def test_no_nan_or_extreme_values(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.zeros(3))
        ctrl = CascadedPIDController()
        states: list[np.ndarray] = []
        takeoff(quad, ctrl, target_alt=12.0, dt=0.005, duration=5.0, states=states)
        arr = np.array(states)
        assert not np.any(np.isnan(arr)), "NaN in takeoff states"
        assert np.all(np.abs(arr[:, :3]) < 50), "Positions exceeded 50m during takeoff"


class TestHover:
    def test_holds_position(self) -> None:
        quad, ctrl = _setup_hovering_quad(np.array([15.0, 15.0, 12.0]))
        target = np.array([15.0, 15.0, 12.0])
        states: list[np.ndarray] = []
        loiter(quad, ctrl, target, dt=0.005, duration=5.0, states=states)
        arr = np.array(states)
        max_drift = np.max(np.linalg.norm(arr[:, :3] - target, axis=1))
        assert max_drift < 0.5, f"Hover drift: {max_drift:.2f} > 0.5 m"

    def test_altitude_stable(self) -> None:
        quad, ctrl = _setup_hovering_quad(np.array([15.0, 15.0, 15.0]))
        states: list[np.ndarray] = []
        loiter(quad, ctrl, np.array([15.0, 15.0, 15.0]), dt=0.005, duration=5.0, states=states)
        arr = np.array(states)
        alt_range = arr[:, 2].max() - arr[:, 2].min()
        assert alt_range < 0.5, f"Altitude oscillation: {alt_range:.2f} > 0.5 m"


class TestPathFollowing:
    def test_straight_line_to_goal(self) -> None:
        quad, ctrl = _setup_hovering_quad(np.array([5.0, 5.0, 12.0]))
        path = np.array([[5, 5, 12], [25, 25, 12.0]])
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(quad, ctrl, path, dt=0.005, pursuit=pp, timeout=40.0, states=states)
        arr = np.array(states)
        dist_to_goal = float(np.linalg.norm(arr[-1, :3] - path[-1]))
        assert dist_to_goal < _GOAL_TOL, f"Distance to goal: {dist_to_goal:.2f}"

    def test_multi_waypoint_path(self) -> None:
        quad, ctrl = _setup_hovering_quad(np.array([3.0, 3.0, 15.0]))
        path = np.array(
            [
                [3, 3, 15],
                [10, 5, 15],
                [15, 15, 15],
                [25, 20, 15],
                [27, 27, 15.0],
            ]
        )
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(quad, ctrl, path, dt=0.005, pursuit=pp, timeout=60.0, states=states)
        arr = np.array(states)
        dist_to_goal = float(np.linalg.norm(arr[-1, :3] - path[-1]))
        assert dist_to_goal < _GOAL_TOL, f"Distance to goal: {dist_to_goal:.2f}"

    def test_no_extreme_tilt(self) -> None:
        quad, ctrl = _setup_hovering_quad(np.array([5.0, 5.0, 12.0]))
        path = np.array([[5, 5, 12], [25, 25, 12.0]])
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(quad, ctrl, path, dt=0.005, pursuit=pp, timeout=40.0, states=states)
        arr = np.array(states)
        max_tilt = np.max(np.abs(arr[:, 3:5]))
        assert max_tilt < np.radians(
            _MAX_TILT_DEG
        ), f"Max tilt: {np.degrees(max_tilt):.1f}° > {_MAX_TILT_DEG}°"

    def test_velocity_stays_reasonable(self) -> None:
        """Drone should never exceed _MAX_VEL during normal path following."""
        quad, ctrl = _setup_hovering_quad(np.array([5.0, 5.0, 15.0]))
        path = np.array([[5, 5, 15], [25, 25, 15.0]])
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(quad, ctrl, path, dt=0.005, pursuit=pp, timeout=40.0, states=states)
        arr = np.array(states)
        speeds = np.linalg.norm(arr[:, 6:9], axis=1)
        max_speed = float(speeds.max())
        assert max_speed < _MAX_VEL, f"Max speed: {max_speed:.2f} m/s > {_MAX_VEL} m/s"

    def test_altitude_maintained_during_xy_flight(self) -> None:
        quad, ctrl = _setup_hovering_quad(np.array([5.0, 5.0, 15.0]))
        path = np.array([[5, 5, 15], [25, 25, 15.0]])
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(quad, ctrl, path, dt=0.005, pursuit=pp, timeout=40.0, states=states)
        arr = np.array(states)
        alt_range = arr[:, 2].max() - arr[:, 2].min()
        assert alt_range < 2.0, f"Altitude range during flight: {alt_range:.2f} > 2.0 m"

    def test_path_tracking_error_bounded(self) -> None:
        """Cross-track error to the path should stay bounded."""
        quad, ctrl = _setup_hovering_quad(np.array([5.0, 5.0, 12.0]))
        path = np.array([[5, 5, 12], [15, 15, 12], [25, 25, 12.0]])
        pp = PurePursuit3D(lookahead=3.0, waypoint_threshold=2.0, adaptive=True)
        states: list[np.ndarray] = []
        fly_path(quad, ctrl, path, dt=0.005, pursuit=pp, timeout=60.0, states=states)
        arr = np.array(states)
        # Compute perpendicular distance to nearest segment
        positions = arr[:, :3]
        max_cross_err = 0.0
        for k in range(len(positions)):
            min_dist = float("inf")
            for seg in range(len(path) - 1):
                a, b = path[seg], path[seg + 1]
                ab = b - a
                ap = positions[k] - a
                t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
                closest = a + t * ab
                min_dist = min(min_dist, float(np.linalg.norm(positions[k] - closest)))
            max_cross_err = max(max_cross_err, min_dist)
        assert max_cross_err < 3.0, f"Max cross-track error: {max_cross_err:.2f} > 3.0 m"


class TestFullMission:
    def test_takeoff_fly_land(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([5.0, 5.0, 0.0]))
        ctrl = CascadedPIDController()
        path = np.array([[5, 5, 12], [20, 20, 12.0]])

        states = fly_mission(
            quad,
            ctrl,
            path,
            cruise_alt=12.0,
            dt=0.005,
            takeoff_duration=4.0,
            landing_duration=5.0,
            loiter_duration=2.0,
        )
        assert not np.any(np.isnan(states)), "NaN in mission states"
        assert states[-1, 2] < 2.0, f"Final altitude: {states[-1, 2]:.2f} > 2.0 m (not landed)"
        assert len(states) > 500, "Mission too short — possible early divergence"

    def test_no_divergence(self) -> None:
        quad = Quadrotor()
        quad.reset(position=np.array([5.0, 5.0, 0.0]))
        ctrl = CascadedPIDController()
        path = np.array([[5, 5, 15], [25, 25, 15.0]])

        states = fly_mission(quad, ctrl, path, cruise_alt=15.0, dt=0.005)
        max_dist = np.max(np.linalg.norm(states[:, :3], axis=1))
        assert max_dist < 50, f"Max distance from origin: {max_dist:.1f} > 50 m"

    def test_mission_velocity_bounded(self) -> None:
        """Velocity should stay bounded through the entire mission lifecycle."""
        quad = Quadrotor()
        quad.reset(position=np.array([5.0, 5.0, 0.0]))
        ctrl = CascadedPIDController()
        path = np.array([[5, 5, 12], [20, 20, 12.0]])

        states = fly_mission(
            quad,
            ctrl,
            path,
            cruise_alt=12.0,
            dt=0.005,
            takeoff_duration=4.0,
            landing_duration=5.0,
            loiter_duration=1.0,
        )
        speeds = np.linalg.norm(states[:, 6:9], axis=1)
        max_speed = float(speeds.max())
        assert max_speed < 6.0, f"Max mission speed: {max_speed:.2f} m/s > 6.0 m/s"
