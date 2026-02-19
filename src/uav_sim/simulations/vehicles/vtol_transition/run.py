# Erwin Lejeune - 2026-02-19
"""VTOL tilt-rotor transition: hover -> cruise -> hover.

Closed-loop PD attitude and altitude controller prevents flipping.
The tilt angle ramps linearly, and the controller adjusts thrust and
torques to maintain stable flight through the transition.

Reference: R. Bapst et al., "Design and Implementation of an Unmanned
Tail-Sitter," IROS, 2015.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.vehicles.vtol import Tiltrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
TARGET_ALT = 15.0


def _pd_attitude(
    state: np.ndarray, des_phi: float, des_theta: float, des_psi: float, inertia: np.ndarray
) -> np.ndarray:
    """Simple PD attitude controller — returns [tau_x, tau_y, tau_z]."""
    phi, theta, psi = state[3], state[4], state[5]
    p, q, r = state[9], state[10], state[11]

    kp_att = 3.0
    kd_att = 1.0

    Ix, Iy, Iz = inertia[0, 0], inertia[1, 1], inertia[2, 2]

    tx = Ix * (kp_att * (des_phi - phi) - kd_att * p)
    ty = Iy * (kp_att * (des_theta - theta) - kd_att * q)

    e_psi = (des_psi - psi + np.pi) % (2 * np.pi) - np.pi
    tz = Iz * (kp_att * 0.5 * e_psi - kd_att * r)

    return np.array([tx, ty, tz])


def main() -> None:
    vtol = Tiltrotor()
    state = np.zeros(12)
    state[:3] = [5.0, 15.0, TARGET_ALT]
    vtol.reset(state=state)

    p = vtol.vtol_params
    m, g = p.mass, p.gravity

    dt, duration = 0.005, 25.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))
    tilt_angles = np.zeros(steps)

    des_vx = 0.0
    kp_alt = 2.0
    kd_alt = 1.5

    for i in range(steps):
        positions[i] = vtol.state[:3]
        eulers[i] = vtol.state[3:6]
        t = i * dt
        z = vtol.state[2]
        vz = vtol.state[8]

        # Tilt schedule: hover -> transition -> cruise -> back -> hover
        if t < 4:
            tilt = 0.0
            des_vx = 0.0
        elif t < 10:
            blend = (t - 4) / 6.0
            tilt = blend * np.pi / 4
            des_vx = blend * 6.0
        elif t < 17:
            tilt = np.pi / 4
            des_vx = 6.0
        elif t < 23:
            blend = (t - 17) / 6.0
            tilt = np.pi / 4 * (1.0 - blend)
            des_vx = 6.0 * (1.0 - blend)
        else:
            tilt = 0.0
            des_vx = 0.0

        tilt_angles[i] = tilt

        ct = np.cos(tilt)
        alt_err = TARGET_ALT - z
        az_cmd = kp_alt * alt_err - kd_alt * vz

        # Thrust along body z-axis must compensate gravity + altitude correction
        T = m * (g + az_cmd) / max(ct, 0.3)
        T = float(np.clip(T, 0.0, m * g * 2.5))

        # Desired pitch: small nose-down to accelerate forward
        vx = vtol.state[6]
        kp_vx = 0.03
        des_theta = -kp_vx * (des_vx - vx) * np.cos(tilt)
        des_theta = float(np.clip(des_theta, -0.2, 0.2))

        tau = _pd_attitude(vtol.state, 0.0, des_theta, 0.0, p.inertia)

        vtol.step(np.array([T, tau[0], tau[1], tau[2], tilt]), dt)

        if np.any(np.isnan(vtol.state)):
            positions = positions[:i]
            eulers = eulers[:i]
            tilt_angles = tilt_angles[:i]
            break

    n_steps = len(positions)
    if n_steps < 2:
        print("VTOL simulation too short — skipping")
        return

    viz = ThreePanelViz(title="VTOL Hover → Cruise Transition", world_size=WORLD_SIZE)

    times = np.arange(n_steps) * dt
    ax_tilt = viz.fig.add_axes([0.60, 0.03, 0.36, 0.18])
    ax_tilt.set_xlim(0, times[-1])
    ax_tilt.set_ylim(-5, 55)
    ax_tilt.set_xlabel("Time [s]", fontsize=7)
    ax_tilt.set_ylabel("Tilt [deg]", fontsize=7)
    ax_tilt.tick_params(labelsize=6)
    ax_tilt.grid(True, alpha=0.2)
    (tilt_line,) = ax_tilt.plot([], [], "darkorange", lw=0.8, label="Tilt")
    (alt_line,) = ax_tilt.plot([], [], "cyan", lw=0.8, label="Alt")
    ax_tilt.legend(fontsize=6, loc="upper right")

    anim = SimAnimator("vtol_transition", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="dodgerblue")

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    vehicle_arts_3d: list = []

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, positions, k)

        clear_vehicle_artists(vehicle_arts_3d)
        R = Quadrotor.rotation_matrix(*eulers[k])
        vehicle_arts_3d.extend(draw_quadrotor_3d(viz.ax3d, positions[k], R, size=1.5))

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dt_t,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=5)
        viz._vehicle_arts_top.append(dt_t)
        clear_vehicle_artists(viz._vehicle_arts_side)
        (dt_s,) = viz.ax_side.plot(positions[k, 0], positions[k, 2], "ko", ms=5, zorder=5)
        viz._vehicle_arts_side.append(dt_s)

        tilt_line.set_data(times[:k], np.degrees(tilt_angles[:k]))
        alt_line.set_data(times[:k], positions[:k, 2])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
