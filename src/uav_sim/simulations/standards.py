"""Shared simulation standards, completion checks, and trajectory references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment.obstacles import Obstacle

from .common import (
    CRUISE_ALT,
    DEFAULT_OMEGA,
    DEFAULT_RX,
    DEFAULT_RY,
    STANDARD_DURATION,
    figure_8_ref,
)
from .common import (
    DEFAULT_ALT as FIG8_ALT,
)


@dataclass(frozen=True)
class SimulationStandard:
    """Single source of truth for mission timing and completion tolerances."""

    dt: float
    duration: float
    takeoff_duration: float
    landing_duration: float
    loiter_duration: float
    lookahead: float
    waypoint_threshold: float
    adaptive: bool
    goal_xy_tol: float
    divergence_pos_bound: float
    stall_window_s: float
    stall_min_progress_m: float
    timeout_multiplier: float
    safety_clearance_m: float
    fallback_policy: Literal["start_goal", "preserve_shape", "none"]

    @property
    def timeout(self) -> float:
        return self.duration * self.timeout_multiplier

    @classmethod
    def flight_coupled(cls) -> "SimulationStandard":
        return cls(
            dt=0.005,
            duration=STANDARD_DURATION,
            takeoff_duration=3.0,
            landing_duration=4.0,
            loiter_duration=1.0,
            lookahead=3.0,
            waypoint_threshold=1.5,
            adaptive=True,
            goal_xy_tol=1.0,
            divergence_pos_bound=500.0,
            stall_window_s=6.0,
            stall_min_progress_m=0.35,
            timeout_multiplier=3.0,
            safety_clearance_m=0.25,
            fallback_policy="start_goal",
        )

    @classmethod
    def estimation_benchmark(cls) -> "SimulationStandard":
        return cls(
            dt=0.005,
            duration=STANDARD_DURATION,
            takeoff_duration=0.0,
            landing_duration=0.0,
            loiter_duration=0.0,
            lookahead=3.0,
            waypoint_threshold=1.5,
            adaptive=True,
            goal_xy_tol=1.0,
            divergence_pos_bound=1_000.0,
            stall_window_s=6.0,
            stall_min_progress_m=0.0,
            timeout_multiplier=1.1,
            safety_clearance_m=0.0,
            fallback_policy="none",
        )

    @classmethod
    def trajectory_tracking(cls) -> "SimulationStandard":
        return cls(
            dt=0.005,
            duration=STANDARD_DURATION,
            takeoff_duration=0.0,
            landing_duration=0.0,
            loiter_duration=0.0,
            lookahead=0.0,
            waypoint_threshold=0.0,
            adaptive=True,
            goal_xy_tol=1.0,
            divergence_pos_bound=500.0,
            stall_window_s=4.0,
            stall_min_progress_m=0.0,
            timeout_multiplier=1.1,
            safety_clearance_m=0.0,
            fallback_policy="none",
        )


@dataclass(frozen=True)
class CompletionStatus:
    goal_reached_xy: bool
    divergence: bool
    stall: bool
    timeout_reason: str
    best_goal_xy_error_m: float
    final_goal_error_m: float

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "goal_reached_xy": self.goal_reached_xy,
            "divergence": self.divergence,
            "stall": self.stall,
            "timeout_reason": self.timeout_reason,
            "best_goal_xy_error_m": self.best_goal_xy_error_m,
            "final_goal_error_m": self.final_goal_error_m,
        }


@dataclass(frozen=True)
class PathSafetyStatus:
    safe: bool
    min_clearance_m: float
    reason: str


def figure_8_reference(
    t: float,
    *,
    rx: float = DEFAULT_RX,
    ry: float = DEFAULT_RY,
    alt: float = FIG8_ALT,
    omega: float = DEFAULT_OMEGA,
    alt_amp: float = 1.5,
    alt_freq: float = 0.3,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return time-parameterized figure-8 reference as (pos, vel, acc)."""
    pos, vel = figure_8_ref(
        t, rx=rx, ry=ry, alt=alt, omega=omega, alt_amp=alt_amp, alt_freq=alt_freq
    )
    acc = np.array(
        [
            -rx * (omega**2) * np.sin(omega * t),
            -ry * ((2.0 * omega) ** 2) * np.sin(2.0 * omega * t),
            -alt_amp * (alt_freq**2) * np.sin(alt_freq * t),
        ]
    )
    return pos, vel, acc


def deterministic_truth_trajectory(
    standard: SimulationStandard,
    *,
    alt: float = CRUISE_ALT,
    rx: float = DEFAULT_RX,
    ry: float = DEFAULT_RY,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build deterministic truth states independent of live flight pursuit."""
    times = np.arange(0.0, standard.duration, standard.dt)
    states = np.zeros((len(times), 12))
    for i, t in enumerate(times):
        ref_p, ref_v, _ = figure_8_reference(t, rx=rx, ry=ry, alt=alt, alt_amp=0.0)
        yaw = float(np.arctan2(ref_v[1], ref_v[0]))
        yaw_rate = 0.0 if i == 0 else (yaw - float(states[i - 1, 5])) / standard.dt
        states[i, :3] = ref_p
        states[i, 5] = yaw
        states[i, 6:9] = ref_v
        states[i, 11] = yaw_rate
    return states, times


def evaluate_path_safety(
    path: NDArray[np.floating],
    obstacles: Sequence[Obstacle],
    *,
    clearance_m: float,
) -> PathSafetyStatus:
    """Check whether a tracking path is collision-free with minimum clearance."""
    if path.size == 0:
        return PathSafetyStatus(False, -1.0, "empty_path")
    min_clearance = float("inf")
    for p in path:
        for obs in obstacles:
            if obs.contains(p):
                return PathSafetyStatus(False, 0.0, "path_collision")
            clearance = float(obs.distance(p))
            min_clearance = min(min_clearance, clearance)
            if clearance < clearance_m:
                return PathSafetyStatus(False, clearance, "clearance_below_threshold")
    if min_clearance == float("inf"):
        min_clearance = 1e9
    return PathSafetyStatus(True, min_clearance, "ok")


def evaluate_completion(
    positions: NDArray[np.floating],
    goal_xyz: NDArray[np.floating],
    *,
    dt: float,
    standard: SimulationStandard,
    completed_tracking: bool = False,
) -> CompletionStatus:
    """Compute mandatory completion flags and error metrics from a mission trace."""
    if positions.size == 0:
        return CompletionStatus(
            goal_reached_xy=False,
            divergence=True,
            stall=False,
            timeout_reason="empty_trace",
            best_goal_xy_error_m=float("inf"),
            final_goal_error_m=float("inf"),
        )

    valid = np.all(np.isfinite(positions), axis=1)
    divergence = bool(
        (not bool(np.all(valid))) or np.any(np.abs(positions) > standard.divergence_pos_bound)
    )

    xy_err = np.linalg.norm(positions[:, :2] - goal_xyz[:2], axis=1)
    best_goal_xy_error_m = float(np.min(xy_err))
    final_goal_error_m = float(np.linalg.norm(positions[-1] - goal_xyz))
    goal_reached = bool(np.any(xy_err <= standard.goal_xy_tol))

    stall = False
    window_steps = max(2, int(standard.stall_window_s / max(dt, 1e-6)))
    if len(positions) > window_steps and standard.stall_min_progress_m > 0.0:
        for i in range(window_steps, len(positions)):
            progress = float(np.linalg.norm(positions[i, :2] - positions[i - window_steps, :2]))
            if progress < standard.stall_min_progress_m and xy_err[i] > standard.goal_xy_tol:
                stall = True
                break

    timeout_reason = "none"
    sim_elapsed = len(positions) * dt
    if divergence:
        timeout_reason = "divergence"
    elif stall:
        timeout_reason = "stall"
    elif not goal_reached and sim_elapsed >= standard.timeout:
        timeout_reason = "timeout"
    elif not goal_reached and not completed_tracking:
        timeout_reason = "path_incomplete"

    return CompletionStatus(
        goal_reached_xy=goal_reached,
        divergence=divergence,
        stall=stall,
        timeout_reason=timeout_reason,
        best_goal_xy_error_m=best_goal_xy_error_m,
        final_goal_error_m=final_goal_error_m,
    )
