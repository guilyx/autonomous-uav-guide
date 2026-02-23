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

_MAX_XY_CMD_DIST = 1.5  # metres — clamp horizontal target for smooth tilt
_MAX_Z_CMD_DIST = 4.0  # metres — vertical can be larger (no tilt impact)


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
    duration: float = 5.0,
    states: list[NDArray] | None = None,
    climb_rate: float = 2.0,
) -> list[NDArray]:
    """Vertical climb to *target_alt* with a smooth altitude ramp."""
    if states is None:
        states = []
    start_z = float(quad.state[2])
    target = quad.state[:3].copy()
    steps = int(duration / dt)
    for i in range(steps):
        if not _is_sane(quad.state):
            break
        states.append(quad.state.copy())
        elapsed = i * dt
        ramp_z = min(target_alt, start_z + climb_rate * elapsed)
        target[2] = ramp_z
        cmd = _limit_target(quad.state[:3], target)
        quad.step(ctrl.compute(quad.state, cmd, dt=dt), dt)
        if abs(quad.state[2] - target_alt) < 0.3 and abs(quad.state[8]) < 0.3:
            break
    return states


def landing(
    quad: Quadrotor,
    ctrl: CascadedPIDController,
    dt: float = 0.005,
    duration: float = 6.0,
    ground_z: float = 0.0,
    states: list[NDArray] | None = None,
    descent_rate: float | None = None,
) -> list[NDArray]:
    """Descend vertically to *ground_z* with a smooth altitude ramp.

    If *descent_rate* is ``None`` it is computed so that the ramp
    reaches ``ground_z`` in roughly 80 % of *duration*, clamped to
    [0.5, 3.0] m/s for realism.
    """
    if states is None:
        states = []
    start_z = float(quad.state[2])
    if descent_rate is None:
        dz = max(start_z - ground_z, 0.1)
        descent_rate = float(np.clip(dz / (duration * 0.8), 0.5, 3.0))
    target = quad.state[:3].copy()
    steps = int(duration / dt)
    for i in range(steps):
        if not _is_sane(quad.state):
            break
        states.append(quad.state.copy())
        if quad.state[2] < ground_z + 0.05:
            break
        elapsed = i * dt
        ramp_z = max(ground_z, start_z - descent_rate * elapsed)
        target[2] = ramp_z
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
    takeoff_duration: float = 5.0,
    landing_duration: float = 6.0,
    loiter_duration: float = 1.0,
    fly_timeout: float = 120.0,
) -> NDArray[np.floating]:
    """Execute a full mission: takeoff -> fly path -> loiter -> land.

    Returns
    -------
    (N, 12) state history array.
    """
    states: list[NDArray] = []
    alt = cruise_alt if cruise_alt is not None else float(np.mean(path[:, 2]))

    init_hover(quad)
    takeoff(quad, ctrl, target_alt=alt, dt=dt, duration=takeoff_duration, states=states)
    fly_path(quad, ctrl, path, dt=dt, pursuit=pursuit, timeout=fly_timeout, states=states)
    loiter(quad, ctrl, path[-1], dt=dt, duration=loiter_duration, states=states)
    landing(quad, ctrl, dt=dt, duration=landing_duration, states=states)

    return np.array(states) if states else np.zeros((1, 12))
