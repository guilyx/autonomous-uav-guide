# Erwin Lejeune - 2026-02-17
"""Standard flight operation primitives for multirotor UAVs.

Provides reusable ``takeoff``, ``landing``, ``loiter``, and ``fly_path``
functions that operate on a :class:`Quadrotor` + :class:`CascadedPIDController`
pair and record full state histories for visualization.

Every function appends to the provided ``states`` list so that the caller
can concatenate a full mission trajectory from sequential operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

_MAX_CMD_DIST = 4.0  # metres — maximum effective distance to the target sent to PID


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


def _limit_target(pos: NDArray, target: NDArray, max_dist: float = _MAX_CMD_DIST) -> NDArray:
    """Clamp effective target to *max_dist* from current position.

    Prevents the low-level PID from seeing large position errors that would
    cause extreme roll/pitch commands on lightweight airframes.
    """
    delta = target - pos
    dist = float(np.linalg.norm(delta))
    if dist <= max_dist or dist < 1e-8:
        return target
    return pos + delta * (max_dist / dist)


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
    for _ in range(max_steps):
        s = quad.state
        if not _is_sane(s):
            break
        states.append(s.copy())
        if pursuit.is_path_complete(s[:3], path):
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
) -> NDArray[np.floating]:
    """Execute a full mission: takeoff -> fly path -> loiter -> land.

    Returns
    -------
    (N, 12) state history array.
    """
    states: list[NDArray] = []
    alt = cruise_alt if cruise_alt is not None else float(np.mean(path[:, 2]))

    takeoff(quad, ctrl, target_alt=alt, dt=dt, duration=takeoff_duration, states=states)
    fly_path(quad, ctrl, path, dt=dt, pursuit=pursuit, states=states)
    loiter(quad, ctrl, path[-1], dt=dt, duration=loiter_duration, states=states)
    landing(quad, ctrl, dt=dt, duration=landing_duration, states=states)

    return np.array(states) if states else np.zeros((1, 12))
