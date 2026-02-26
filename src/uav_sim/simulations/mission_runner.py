"""Standard mission runner for flight-coupled simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import BoxObstacle, Obstacle
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.path_tracking.flight_ops import fly_mission_trace
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
    tracking_end_idx: int
    tracking_fallback: bool
    fallback_reason: str
    path_min_clearance_m: float


def _segment_safe(
    start: NDArray[np.floating],
    end: NDArray[np.floating],
    obstacles: Sequence[Obstacle],
    clearance_m: float,
    sample_spacing: float = 0.5,
) -> bool:
    """Check safety along a segment using sampled points."""
    d = end - start
    dist = float(np.linalg.norm(d))
    n = max(2, int(np.ceil(dist / max(sample_spacing, 1e-6))) + 1)
    for a in np.linspace(0.0, 1.0, n):
        p = (1.0 - a) * start + a * end
        for obs in obstacles:
            if obs.contains(p):
                return False
            if float(obs.distance(p)) < clearance_m:
                return False
    return True


def _safe_fallback_path(
    planned_path: NDArray[np.floating],
    obstacles: Sequence[Obstacle],
    clearance_m: float,
    *,
    preserve_shape: bool = False,
) -> NDArray[np.floating]:
    """Build a conservative fallback path when the nominal path is unsafe."""
    if len(planned_path) == 0:
        return planned_path

    box_obstacles = [obs for obs in obstacles if isinstance(obs, BoxObstacle)]
    if preserve_shape and len(box_obstacles) == len(obstacles):
        key_path = rdp_simplify(planned_path, epsilon=0.75)
        if len(key_path) < 2:
            key_path = planned_path[[0, -1]]
        max_coord = max(
            float(np.max(planned_path)),
            max(float(np.max(o.max_corner)) for o in box_obstacles),
        )
        world_size = int(max(30, np.ceil(max_coord + 4.0)))
        merged: list[NDArray[np.floating]] = [key_path[0]]
        for i in range(1, len(key_path)):
            s = merged[-1]
            g = key_path[i]
            if _segment_safe(s, g, obstacles, clearance_m):
                merged.append(g.copy())
                continue
            local = plan_through_obstacles(
                box_obstacles,
                start=s,
                goal=g,
                world_size=world_size,
                inflate=max(1, int(np.ceil(clearance_m))),
                smooth_epsilon=1.0,
                smooth_spacing=0.8,
            )
            if local is None or len(local) < 2:
                return planned_path
            merged.extend(p.copy() for p in local[1:])
        return np.array(merged)

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
    fallback_policy: Literal["start_goal", "preserve_shape", "none"] | None = None,
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
    policy = fallback_policy or standard.fallback_policy

    if obstacles is not None and len(planned_path) > 0:
        safety = evaluate_path_safety(
            planned_path, obstacles, clearance_m=standard.safety_clearance_m
        )
        min_clearance = safety.min_clearance_m
        if not safety.safe:
            tracking_fallback = True
            fallback_reason = safety.reason
            if policy == "none":
                fallback_reason = f"{fallback_reason}_fallback_disabled"
            else:
                if fallback_path is None:
                    fallback_path = _safe_fallback_path(
                        planned_path,
                        obstacles,
                        clearance_m=standard.safety_clearance_m,
                        preserve_shape=(policy == "preserve_shape"),
                    )
                safe_path = fallback_path
                fb_safety = evaluate_path_safety(
                    safe_path, obstacles, clearance_m=standard.safety_clearance_m
                )
                min_clearance = min(min_clearance, fb_safety.min_clearance_m)
                if not fb_safety.safe:
                    safe_path = planned_path
                    fallback_reason = f"{fallback_reason}_{policy}_failed"
                else:
                    fallback_reason = f"{fallback_reason}_{policy}"

    trace = fly_mission_trace(
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
    states = trace.states
    positions = states[:, :3]
    path_complete = (
        bool(pursuit.is_path_complete(positions[trace.tracking_end_idx], safe_path))
        if len(safe_path) > 0
        else False
    )
    completion_positions = positions[: trace.tracking_end_idx + 1]
    completion = evaluate_completion(
        completion_positions,
        safe_path[-1] if len(safe_path) > 0 else quad.state[:3],
        dt=standard.dt,
        standard=standard,
        completed_tracking=path_complete,
    )
    if min_clearance == float("inf"):
        min_clearance = 1e9
    return MissionResult(
        states=states,
        tracking_path=safe_path,
        planned_path=planned_path,
        completion=completion,
        path_complete=path_complete,
        tracking_end_idx=trace.tracking_end_idx,
        tracking_fallback=tracking_fallback,
        fallback_reason=fallback_reason,
        path_min_clearance_m=min_clearance,
    )
