# Erwin Lejeune - 2026-02-19
"""Fixed-wing level flight and gentle banked turn with closed-loop autopilot.

PD controllers maintain altitude (via pitch/throttle), heading (via
coordinated turn), and wings-level (via roll). Shows the aircraft
performing straight level flight then a wide banked turn.

Reference: R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and
Practice," Princeton University Press, 2012.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.vehicles.fixed_wing import FixedWing
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_fixed_wing_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
TARGET_ALT = 15.0
TARGET_SPEED = 8.0


def main() -> None:
    fw = FixedWing()
    state = np.zeros(12)
    state[:3] = [3.0, 15.0, TARGET_ALT]
    state[6] = TARGET_SPEED  # initial forward speed
    fw.reset(state=state)

    dt, duration = 0.002, 18.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))

    # PD gains
    kp_pitch, kd_pitch = 0.8, 0.3
    kp_roll, kd_roll = 1.0, 0.2
    kp_yaw, kd_yaw = 0.3, 0.1
    kp_alt, kd_alt = 0.08, 0.04
    kp_spd = 0.05

    des_heading = 0.0  # initial: fly in +x direction

    for i in range(steps):
        positions[i] = fw.state[:3]
        eulers[i] = fw.state[3:6]
        t = i * dt

        phi, theta, psi = fw.state[3], fw.state[4], fw.state[5]
        pb, qb, rb = fw.state[9], fw.state[10], fw.state[11]
        u_body = fw.state[6]
        z = fw.state[2]
        vz = fw.state[8]

        # Heading schedule: straight(0-5) -> right turn(5-12) -> level out(12+)
        if t < 5:
            des_heading = 0.0
        elif t < 12:
            des_heading = min((t - 5) * 0.15, 1.2)
        else:
            des_heading = 1.2

        # Altitude hold → desired pitch
        alt_err = TARGET_ALT - z
        des_theta = kp_alt * alt_err - kd_alt * vz
        des_theta = float(np.clip(des_theta, -0.15, 0.15))

        # Heading hold → desired roll (coordinated turn)
        heading_err = (des_heading - psi + np.pi) % (2 * np.pi) - np.pi
        des_phi = kp_yaw * heading_err
        des_phi = float(np.clip(des_phi, -0.4, 0.4))

        # Pitch PD → elevator
        elevator = kp_pitch * (des_theta - theta) - kd_pitch * qb
        elevator = float(np.clip(elevator, -0.3, 0.3))

        # Roll PD → aileron
        aileron = kp_roll * (des_phi - phi) - kd_roll * pb
        aileron = float(np.clip(aileron, -0.3, 0.3))

        # Yaw damper → rudder (reduce sideslip)
        rudder = -kd_yaw * rb
        rudder = float(np.clip(rudder, -0.2, 0.2))

        # Speed hold → throttle
        speed_err = TARGET_SPEED - u_body
        throttle = 0.35 + kp_spd * speed_err
        throttle = float(np.clip(throttle, 0.1, 0.8))

        fw.step(np.array([elevator, aileron, rudder, throttle]), dt)

        if np.any(np.isnan(fw.state)):
            positions = positions[:i]
            eulers = eulers[:i]
            break

        pos = fw.state[:3]
        if np.any(pos < -5) or np.any(pos > WORLD_SIZE + 5):
            positions = positions[: i + 1]
            eulers = eulers[: i + 1]
            break

    n_steps = len(positions)
    if n_steps < 2:
        print("Fixed-wing simulation too short — skipping")
        return

    viz = ThreePanelViz(title="Fixed-Wing Flight — Autopilot", world_size=WORLD_SIZE)

    # Data inset: altitude + speed
    times = np.arange(n_steps) * dt
    ax_data = viz.fig.add_axes([0.60, 0.03, 0.36, 0.18])
    ax_data.set_xlim(0, times[-1])
    ax_data.set_ylim(0, 25)
    ax_data.set_xlabel("Time [s]", fontsize=7)
    ax_data.tick_params(labelsize=6)
    ax_data.grid(True, alpha=0.2)
    (alt_line,) = ax_data.plot([], [], "cyan", lw=0.8, label="Altitude [m]")
    (spd_line,) = ax_data.plot([], [], "orange", lw=0.8, label="Speed [m/s]")
    ax_data.axhline(TARGET_ALT, color="cyan", lw=0.5, ls="--", alpha=0.5)
    ax_data.axhline(TARGET_SPEED, color="orange", lw=0.5, ls="--", alpha=0.5)
    ax_data.legend(fontsize=6, loc="upper right")

    anim = SimAnimator("fixed_wing_flight", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="royalblue")

    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    vehicle_arts_3d: list = []

    speeds = np.sqrt(np.sum(np.diff(positions, axis=0, prepend=positions[:1]) ** 2, axis=1)) / dt

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, positions, k)

        clear_vehicle_artists(vehicle_arts_3d)
        R = Quadrotor.rotation_matrix(*eulers[k])
        vehicle_arts_3d.extend(draw_fixed_wing_3d(viz.ax3d, positions[k], R, scale=3.0))

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dot_top,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=5)
        viz._vehicle_arts_top.append(dot_top)
        clear_vehicle_artists(viz._vehicle_arts_side)
        (dot_side,) = viz.ax_side.plot(positions[k, 0], positions[k, 2], "ko", ms=5, zorder=5)
        viz._vehicle_arts_side.append(dot_side)

        alt_line.set_data(times[:k], positions[:k, 2])
        spd_line.set_data(times[:k], speeds[:k])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
