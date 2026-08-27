# Erwin Lejeune - 2026-08-27
"""Obstacle slalom: a reckless controller made safe by a barrier.

The nominal controller flies dead straight at the goal and has never heard
of the obstacles. Everything keeping the vehicle out of them is the barrier,
which is the point: the safety argument does not depend on the controller
being any good.

Run twice, with the filter bypassed for comparison, so the same reckless
command can be seen going through a pillar and then around it.

Reference: A. D. Ames et al., "Control Barrier Function Based Quadratic
Programs for Safety Critical Systems," IEEE TAC, 2017.
DOI: 10.1109/TAC.2016.2638961
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flybots.logging import SimLogger
from flybots.safety import SafetyFilter, SpeedLimitBarrier, SphereObstacleBarrier
from flybots.visualization import SimAnimator, attitude_from_velocity
from flybots.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

WORLD = 60.0
ALT = 15.0
RADIUS = 4.0
CLEARANCE = 1.2
MAX_SPEED = 7.0
MAX_ACCEL = 9.0
DT = 0.02
STEPS = 2600

START = np.array([[5.0, 30.0, ALT]])
GOAL = np.array([[55.0, 30.0, ALT]])
# Staggered either side of the straight line, so the vehicle has to weave
# rather than sidestep once -- and offset by less than a radius, so the
# straight-line path goes *through* each one rather than grazing it. At a
# four-metre offset the line is exactly tangent, and the unfiltered run
# looks almost safe by accident.
CENTRES = np.array(
    [
        [18.0, 28.0, ALT],
        [30.0, 32.0, ALT],
        [42.0, 28.0, ALT],
    ]
)


def _fly(use_filter: bool):
    barrier = SphereObstacleBarrier(CENTRES, RADIUS, clearance=CLEARANCE, k1=4.0, k2=4.0)
    filt = SafetyFilter([barrier, SpeedLimitBarrier(MAX_SPEED)], u_min=-MAX_ACCEL, u_max=MAX_ACCEL)

    pos = START.copy()
    vel = np.zeros_like(pos)
    history = np.zeros((STEPS, 3))
    clearance = np.zeros(STEPS)
    goal_err = np.zeros(STEPS)

    for step in range(STEPS):
        nominal = np.clip(2.2 * (GOAL - pos) - 3.0 * vel, -MAX_ACCEL, MAX_ACCEL)
        command = filt(pos, vel, nominal).command if use_filter else nominal
        vel = vel + command * DT
        pos = pos + vel * DT

        history[step] = pos[0]
        surface = np.linalg.norm(pos[0] - CENTRES, axis=1) - RADIUS
        clearance[step] = float(surface.min())
        goal_err[step] = float(np.linalg.norm(pos[0] - GOAL[0]))

    return history, clearance, goal_err


def main() -> None:
    safe, clear_on, err_on = _fly(True)
    reckless, clear_off, err_off = _fly(False)
    times = np.arange(STEPS) * DT

    logger = SimLogger("obstacle_slalom", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "CBF obstacle avoidance")
    logger.log_metadata("obstacle_radius_m", RADIUS)
    logger.log_metadata("clearance_m", CLEARANCE)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            position=safe[step],
            surface_clearance=float(clear_on[step]),
            surface_clearance_unfiltered=float(clear_off[step]),
            goal_error=float(err_on[step]),
        )
    logger.log_summary("min_clearance_m", float(clear_on.min()))
    logger.log_summary("min_clearance_unfiltered_m", float(clear_off.min()))
    logger.log_summary("required_clearance_m", CLEARANCE)
    logger.log_summary("goal_reached", bool(err_on[-1] < 1.0))
    logger.save()

    skip = max(1, STEPS // 140)
    idx = list(range(0, STEPS, skip))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_clr = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    fig.suptitle("Obstacle Slalom — the controller is reckless, the barrier is not", fontsize=13)

    ax3d.set_xlim(0, WORLD)
    ax3d.set_ylim(0, WORLD)
    ax3d.set_zlim(0, 30)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    theta = np.linspace(0, 2 * np.pi, 40)
    for centre in CENTRES:
        ax3d.plot(
            centre[0] + RADIUS * np.cos(theta),
            centre[1] + RADIUS * np.sin(theta),
            np.full_like(theta, ALT),
            color="crimson",
            lw=1.6,
        )
        ax3d.plot(
            centre[0] + (RADIUS + CLEARANCE) * np.cos(theta),
            centre[1] + (RADIUS + CLEARANCE) * np.sin(theta),
            np.full_like(theta, ALT),
            color="crimson",
            lw=0.8,
            ls="--",
            alpha=0.6,
        )
    ax3d.plot(reckless[:, 0], reckless[:, 1], reckless[:, 2], color="tab:red", lw=1.0, alpha=0.7)
    ax3d.scatter(*GOAL[0], color="lime", s=90, marker="*")

    ax_clr.set_xlim(0, times[-1])
    ax_clr.set_ylim(-RADIUS, 18)
    ax_clr.set_xlabel("Time [s]", fontsize=8)
    ax_clr.set_ylabel("Clearance to surface [m]", fontsize=8)
    ax_clr.set_title("Negative means inside the obstacle", fontsize=9)
    ax_clr.grid(True, alpha=0.3)
    ax_clr.axhline(0.0, color="k", lw=1.0)
    ax_clr.axhline(CLEARANCE, color="crimson", ls="--", lw=1.1, label="required clearance")
    (l_on,) = ax_clr.plot([], [], color="tab:green", lw=1.5, label="filtered")
    (l_off,) = ax_clr.plot([], [], color="tab:red", lw=1.2, label="unfiltered")
    ax_clr.legend(fontsize=7, loc="upper right")

    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(err_on.max(), err_off.max()) * 1.1)
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Distance to goal [m]", fontsize=8)
    ax_err.set_title("Safety did not cost the goal here", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    (e_on,) = ax_err.plot([], [], color="tab:green", lw=1.5, label="filtered")
    (e_off,) = ax_err.plot([], [], color="tab:red", lw=1.2, label="unfiltered")
    ax_err.legend(fontsize=7, loc="upper right")

    (trail,) = ax3d.plot([], [], [], color="tab:green", lw=1.8)
    veh: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("obstacle_slalom", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        trail.set_data(safe[: step + 1, 0], safe[: step + 1, 1])
        trail.set_3d_properties(safe[: step + 1, 2])

        clear_vehicle_artists(veh)
        v = (safe[step] - safe[step - 1]) / DT if step > 0 else np.zeros(3)
        veh.extend(draw_quadrotor_3d(ax3d, safe[step], attitude_from_velocity(v), size=1.6))

        l_on.set_data(times[:step], clear_on[:step])
        l_off.set_data(times[:step], clear_off[:step])
        e_on.set_data(times[:step], err_on[:step])
        e_off.set_data(times[:step], err_off[:step])
        title.set_text(f"t = {step * DT:.1f} s   clearance {clear_on[step]:.2f} m")

    anim.animate(update, len(idx))
    anim.save()

    print(f"  filtered   min clearance {clear_on.min():+.3f} m   goal error {err_on[-1]:.2f} m")
    print(f"  unfiltered min clearance {clear_off.min():+.3f} m   goal error {err_off[-1]:.2f} m")
    print(f"  required   {CLEARANCE:.3f} m")


if __name__ == "__main__":
    main()
