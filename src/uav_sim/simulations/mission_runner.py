"""Standard mission runner for flight-coupled simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import BoxObstacle, Obstacle
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import rdp_simplify
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

from .standards import (
    CompletionStatus,
    SimulationStandard,
    evaluate_completion,
    evaluate_path_safety,
)


@dataclass(frozen=True)
class MissionResult:
    states: NDArray[np.floating]
    tracking_path: NDArray[np.floating]
    planned_path: NDArray[np.floating]
    completion: CompletionStatus
    path_complete: bool
    tracking_fallback: bool
    fallback_reason: str
    path_min_clearance_m: float


def _safe_fallback_path(
    planned_path: NDArray[np.floating],
    obstacles: Sequence[Obstacle],
    clearance_m: float,
) -> NDArray[np.floating]:
    """Build a conservative fallback path when the nominal path is unsafe."""
    if len(planned_path) == 0:
        return planned_path

    box_obstacles = [obs for obs in obstacles if isinstance(obs, BoxObstacle)]
    if len(box_obstacles) == len(obstacles):
        max_coord = max(
            float(np.max(planned_path)),
            max(float(np.max(o.max_corner)) for o in box_obstacles),
        )
        world_size = int(max(30, np.ceil(max_coord + 4.0)))
        astar = plan_through_obstacles(
            box_obstacles,
            start=planned_path[0],
            goal=planned_path[-1],
            world_size=world_size,
            inflate=max(1, int(np.ceil(clearance_m))),
            smooth_epsilon=1.5,
            smooth_spacing=1.0,
        )
        if astar is not None and len(astar) >= 2:
            return astar

        max_obs_z = max(float(o.max_corner[2]) for o in box_obstacles)
        safe_alt = max(float(np.max(planned_path[:, 2])), max_obs_z + clearance_m + 1.0)
        start = planned_path[0].copy()
        goal = planned_path[-1].copy()
        mid_start = np.array([start[0], start[1], safe_alt], dtype=float)
        mid_goal = np.array([goal[0], goal[1], safe_alt], dtype=float)
        return np.vstack([start, mid_start, mid_goal, goal])

    # Generic fallback when obstacle models are not box-based.
    return rdp_simplify(planned_path, epsilon=0.5)


def run_standard_mission(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    planned_path: NDArray[np.floating],
    *,
    standard: SimulationStandard,
    obstacles: Sequence[Obstacle] | None = None,
    fallback_path: NDArray[np.floating] | None = None,
) -> MissionResult:
    """Run takeoff->tracking->loiter->landing with unified completion checks."""
    pursuit = PurePursuit3D(
        lookahead=standard.lookahead,
        waypoint_threshold=standard.waypoint_threshold,
        adaptive=standard.adaptive,
    )
    safe_path = planned_path
    tracking_fallback = False
    fallback_reason = "none"
    min_clearance = float("inf")

    if obstacles is not None and len(planned_path) > 0:
        safety = evaluate_path_safety(
            planned_path, obstacles, clearance_m=standard.safety_clearance_m
        )
        min_clearance = safety.min_clearance_m
        if not safety.safe:
            tracking_fallback = True
            fallback_reason = safety.reason
            if fallback_path is None:
                fallback_path = _safe_fallback_path(
                    planned_path,
                    obstacles,
                    clearance_m=standard.safety_clearance_m,
                )
            safe_path = fallback_path
            fb_safety = evaluate_path_safety(
                safe_path, obstacles, clearance_m=standard.safety_clearance_m
            )
            min_clearance = min(min_clearance, fb_safety.min_clearance_m)
            if not fb_safety.safe:
                fallback_reason = f"{fallback_reason}_fallback_unsafe"

    states = fly_mission(
        quad,
        ctrl,
        safe_path,
        cruise_alt=float(np.mean(safe_path[:, 2])) if len(safe_path) > 0 else float(quad.state[2]),
        dt=standard.dt,
        pursuit=pursuit,
        takeoff_duration=standard.takeoff_duration,
        landing_duration=standard.landing_duration,
        loiter_duration=standard.loiter_duration,
        fly_timeout=standard.timeout,
        stall_window_s=standard.stall_window_s,
        stall_min_progress_m=standard.stall_min_progress_m,
    )
    positions = states[:, :3]
    path_complete = (
        bool(pursuit.is_path_complete(positions[-1], safe_path)) if len(safe_path) > 0 else False
    )
    completion = evaluate_completion(
        positions,
        safe_path[-1] if len(safe_path) > 0 else quad.state[:3],
        dt=standard.dt,
        standard=standard,
        completed_tracking=path_complete,
    )
    return MissionResult(
        states=states,
        tracking_path=safe_path,
        planned_path=planned_path,
        completion=completion,
        path_complete=path_complete,
        tracking_fallback=tracking_fallback,
        fallback_reason=fallback_reason,
        path_min_clearance_m=min_clearance,
    )
