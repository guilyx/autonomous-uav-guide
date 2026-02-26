# Erwin Lejeune - 2026-02-17
"""Standard flight operation primitives for multirotor UAVs.

Provides reusable ``takeoff``, ``landing``, ``loiter``, and ``fly_path``
functions that operate on a :class:`Quadrotor` + :class:`CascadedPIDController`
pair and record full state histories for visualization.

Every function appends to the provided ``states`` list so that the caller
can concatenate a full mission trajectory from sequential operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

_MAX_XY_CMD_DIST = 1.5  # metres — clamp horizontal target for smooth tilt
_MAX_Z_CMD_DIST = 4.0  # metres — vertical can be larger (no tilt impact)


@dataclass(frozen=True)
class MissionTrace:
    """Mission trajectory plus phase boundaries in the concatenated trace."""

    states: NDArray[np.floating]
    takeoff_end_idx: int
    tracking_end_idx: int
    loiter_end_idx: int
    landing_end_idx: int


def init_hover(quad: Quadrotor) -> None:
    """Pre-spin motors to hover speed for a clean start at altitude.

    Call before :func:`fly_path` when the quadrotor is already at cruise
    altitude and motors have not been spun up yet.  Without this, the
    zero-speed motors cause a brief altitude drop while the first-order
    lag catches up.
    """
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))


def _is_sane(state: NDArray) -> bool:
    """Return False if state has NaN or is wildly out of range."""
    return bool(not np.any(np.isnan(state[:3])) and np.all(np.abs(state[:3]) < 500))


def _limit_target(
    pos: NDArray,
    target: NDArray,
    max_xy: float = _MAX_XY_CMD_DIST,
    max_z: float = _MAX_Z_CMD_DIST,
) -> NDArray:
    """Clamp effective target separately for XY and Z.

    Horizontal clamping prevents the PID from demanding extreme tilt;
    vertical clamping can be looser because thrust adjustments don't
    affect roll/pitch.
    """
    out = target.copy()
    dxy = target[:2] - pos[:2]
    dist_xy = float(np.linalg.norm(dxy))
    if dist_xy > max_xy and dist_xy > 1e-8:
        out[:2] = pos[:2] + dxy * (max_xy / dist_xy)

    dz = target[2] - pos[2]
    if abs(dz) > max_z:
        out[2] = pos[2] + np.sign(dz) * max_z
    return out


def takeoff(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    target_alt: float,
    dt: float = 0.005,
    duration: float = 3.0,
    states: list[NDArray] | None = None,
) -> list[NDArray]:
    """Vertical climb to *target_alt* from current position."""
    if states is None:
        states = []
    target = quad.state[:3].copy()
    target[2] = target_alt
    steps = int(duration / dt)
    for _ in range(steps):
        if not _is_sane(quad.state):
            break
        states.append(quad.state.copy())
        cmd = _limit_target(quad.state[:3], target)
        quad.step(ctrl.compute(quad.state, cmd, dt=dt), dt)
    return states


def landing(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    dt: float = 0.005,
    duration: float = 4.0,
    ground_z: float = 0.0,
    states: list[NDArray] | None = None,
) -> list[NDArray]:
    """Descend vertically to *ground_z* from current position."""
    if states is None:
        states = []
    target = quad.state[:3].copy()
    target[2] = ground_z
    steps = int(duration / dt)
    for _ in range(steps):
        if not _is_sane(quad.state):
            break
        states.append(quad.state.copy())
        if quad.state[2] < ground_z + 0.05:
            break
        cmd = _limit_target(quad.state[:3], target)
        quad.step(ctrl.compute(quad.state, cmd, dt=dt), dt)
    return states


def loiter(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    position: NDArray[np.floating],
    dt: float = 0.005,
    duration: float = 2.0,
    states: list[NDArray] | None = None,
) -> list[NDArray]:
    """Hold position at *position* for *duration* seconds."""
    if states is None:
        states = []
    steps = int(duration / dt)
    for _ in range(steps):
        if not _is_sane(quad.state):
            break
        states.append(quad.state.copy())
        cmd = _limit_target(quad.state[:3], position)
        quad.step(ctrl.compute(quad.state, cmd, dt=dt), dt)
    return states


def fly_path(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    path: NDArray[np.floating],
    dt: float = 0.005,
    pursuit: PurePursuit3D | None = None,
    timeout: float = 30.0,
    stall_window_s: float = 0.0,
    stall_min_progress_m: float = 0.0,
    states: list[NDArray] | None = None,
) -> list[NDArray]:
    """Fly along *path* using pure pursuit, recording states.

    Parameters
    ----------
    quad, ctrl : vehicle + low-level controller.
    path : (N, 3) waypoint array.
    dt : simulation time step.
    pursuit : optional PurePursuit3D instance (created with defaults if None).
    timeout : maximum flight time [s].
    states : list to append to (created if None).
    """
    if states is None:
        states = []
    if pursuit is None:
        pursuit = PurePursuit3D(lookahead=1.0, waypoint_threshold=0.5, adaptive=True)

    pursuit.reset()
    max_steps = int(timeout / dt)
    window_steps = max(2, int(stall_window_s / max(dt, 1e-6)))
    for _ in range(max_steps):
        s = quad.state
        if not _is_sane(s):
            break
        states.append(s.copy())
        if pursuit.is_path_complete(s[:3], path):
            break
        if stall_window_s > 0.0 and stall_min_progress_m > 0.0 and len(states) > window_steps:
            progress = float(np.linalg.norm(states[-1][:2] - states[-window_steps][:2]))
            goal_xy = float(np.linalg.norm(s[:2] - path[-1, :2]))
            if progress < stall_min_progress_m and goal_xy > pursuit.waypoint_threshold:
                break
        vel = s[6:9] if len(s) >= 9 else None
        target = pursuit.compute_target(s[:3], path, velocity=vel)
        cmd = _limit_target(s[:3], target)
        quad.step(ctrl.compute(s, cmd, dt=dt), dt)
    return states


def fly_mission(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    path: NDArray[np.floating],
    cruise_alt: float | None = None,
    dt: float = 0.005,
    pursuit: PurePursuit3D | None = None,
    takeoff_duration: float = 3.0,
    landing_duration: float = 4.0,
    loiter_duration: float = 1.0,
    fly_timeout: float = 120.0,
    stall_window_s: float = 0.0,
    stall_min_progress_m: float = 0.0,
) -> NDArray[np.floating]:
    """Execute a full mission: takeoff -> fly path -> loiter -> land.

    Returns
    -------
    (N, 12) state history array.
    """
    return fly_mission_trace(
        quad,
        ctrl,
        path,
        cruise_alt=cruise_alt,
        dt=dt,
        pursuit=pursuit,
        takeoff_duration=takeoff_duration,
        landing_duration=landing_duration,
        loiter_duration=loiter_duration,
        fly_timeout=fly_timeout,
        stall_window_s=stall_window_s,
        stall_min_progress_m=stall_min_progress_m,
    ).states


def fly_mission_trace(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    path: NDArray[np.floating],
    cruise_alt: float | None = None,
    dt: float = 0.005,
    pursuit: PurePursuit3D | None = None,
    takeoff_duration: float = 3.0,
    landing_duration: float = 4.0,
    loiter_duration: float = 1.0,
    fly_timeout: float = 120.0,
    stall_window_s: float = 0.0,
    stall_min_progress_m: float = 0.0,
) -> MissionTrace:
    """Execute a full mission and return phase boundaries."""
    states: list[NDArray] = []
    alt = cruise_alt if cruise_alt is not None else float(np.mean(path[:, 2]))

    init_hover(quad)
    takeoff(quad, ctrl, target_alt=alt, dt=dt, duration=takeoff_duration, states=states)
    takeoff_end_idx = len(states) - 1

    fly_path(
        quad,
        ctrl,
        path,
        dt=dt,
        pursuit=pursuit,
        timeout=fly_timeout,
        stall_window_s=stall_window_s,
        stall_min_progress_m=stall_min_progress_m,
        states=states,
    )
    tracking_end_idx = len(states) - 1

    loiter(quad, ctrl, path[-1], dt=dt, duration=loiter_duration, states=states)
    loiter_end_idx = len(states) - 1

    landing(quad, ctrl, dt=dt, duration=landing_duration, states=states)
    landing_end_idx = len(states) - 1

    trace = np.array(states) if states else np.zeros((1, 12))
    return MissionTrace(
        states=trace,
        takeoff_end_idx=max(0, takeoff_end_idx),
        tracking_end_idx=max(0, tracking_end_idx),
        loiter_end_idx=max(0, loiter_end_idx),
        landing_end_idx=max(0, landing_end_idx),
    )
