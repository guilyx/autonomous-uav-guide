# Erwin Lejeune - 2026-08-27
"""Connectivity annulus: two barriers that pull opposite ways.

`SafeDistanceBarrier` pushes every pair apart; `ConnectivityBarrier` pulls
every pair together. Run them at once and the fleet is squeezed into a
shell -- close enough to talk, far enough not to touch -- without anything
having planned that shape. It falls out of the intersection of two safe
sets.

The radio range then **breathes** -- degrading and recovering on a slow
cycle, as a real link does when terrain or weather moves against it -- and
the shell breathes with it. The fleet expands to fill whatever range it is
given and contracts as that range collapses, never once closing inside the
safe distance. Nothing schedules that; the barrier is simply time-varying
and the QP re-solves against it every step.

Two earlier versions of this scene are worth recording. Pushing outward
against a fixed range reaches equilibrium in about thirteen seconds and
then nothing moves for the rest of the run. Flying the whole formation
along a figure-8 instead put it in motion but pulled it together: the
guide attraction beat the outward push, the furthest pair collapsed to
half the range, the connectivity barrier stopped being active at all, and
the shell -- the entire point -- quietly disappeared.

This is also the honest way to show a filter failing. Widen the safe
distance past the communication range and the two barriers ask for
something impossible: the QP has no solution, and the filter says so rather
than quietly returning an unsafe command. The comparison run does that.

References:
- A. D. Ames et al., "Control Barrier Function Based Quadratic Programs for
  Safety Critical Systems," IEEE TAC, 2017. DOI: 10.1109/TAC.2016.2638961
- U. Borrmann et al., "Control Barrier Certificates for Safe Swarm
  Behavior," IFAC ADHS, 2015. DOI: 10.1016/j.ifacol.2015.11.154
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flybots.logging import SimLogger
from flybots.safety import (
    ConnectivityBarrier,
    SafeDistanceBarrier,
    SafetyFilter,
    SpeedLimitBarrier,
)
from flybots.visualization import SimAnimator, attitude_from_velocity
from flybots.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

WORLD = 70.0
ALT = 35.0
N_AGENTS = 6
# The radio degrades and recovers on a slow cycle; the shell follows it.
RANGE_MIN = 12.0
RANGE_MAX = 32.0
RANGE_OMEGA = 0.30
SAFE_DISTANCE = 6.0
COMM_RANGE = RANGE_MAX
MAX_SPEED = 6.0
MAX_ACCEL = 8.0
DT = 0.02
STEPS = 2200


def _fly(safe_distance: float, comm_range: float):
    # Start *inside* the safe set. A CBF guarantees forward invariance --
    # it holds a safe set you are already in -- and cannot undo a violated
    # initial condition. Seeding at random put three pairs inside the safe
    # distance at t=0, and the run then reported that as its minimum, which
    # reads as a barrier failure when it is nothing of the kind.
    # On a circle of radius R the closest pair is 2R·sin(pi/N) and the
    # furthest is the diameter 2R, so R has to clear the safe distance *and*
    # keep the diameter inside the comm range. The first attempt here sized
    # the ring off the safe distance alone and put diametric pairs at 28 m
    # against a 26 m radio -- satisfying one barrier by breaking the other.
    rng = np.random.default_rng(5)
    angles = np.linspace(0.0, 2.0 * np.pi, N_AGENTS, endpoint=False)
    ring = 0.40 * comm_range
    pos = np.column_stack(
        [
            WORLD / 2 + ring * np.cos(angles),
            WORLD / 2 + ring * np.sin(angles),
            np.full(N_AGENTS, ALT),
        ]
    )
    pos[:, :2] += rng.uniform(-0.4, 0.4, (N_AGENTS, 2))
    vel = np.zeros_like(pos)

    connectivity = ConnectivityBarrier(comm_range, k1=4.0, k2=4.0)
    filt = SafetyFilter(
        [
            SafeDistanceBarrier(safe_distance, k1=4.0, k2=4.0),
            connectivity,
            SpeedLimitBarrier(MAX_SPEED),
        ],
        u_min=-MAX_ACCEL,
        u_max=MAX_ACCEL,
    )

    history = np.zeros((STEPS, N_AGENTS, 3))
    range_hist = np.zeros(STEPS)
    closest = np.zeros(STEPS)
    furthest = np.zeros(STEPS)
    infeasible = np.zeros(STEPS)

    for step in range(STEPS):
        t = step * DT
        # A link budget that comes and goes. The connectivity barrier is
        # rebuilt against it each step, which is all a time-varying safe set
        # requires -- the QP does not care that the bound moved.
        live_range = 0.5 * (comm_range + RANGE_MIN) + 0.5 * (comm_range - RANGE_MIN) * np.sin(
            RANGE_OMEGA * t
        )
        connectivity.comm_range = float(live_range)
        range_hist[step] = live_range

        # The only task is "spread out". Every bit of structure in the
        # result is the barriers'.
        radial = pos - np.array([WORLD / 2, WORLD / 2, ALT])
        radial[:, 2] = 0.0
        norm = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
        nominal = np.clip(6.0 * radial / norm - 1.2 * vel, -MAX_ACCEL, MAX_ACCEL)

        report = filt(pos, vel, nominal)
        infeasible[step] = float(report.infeasible)
        vel = vel + report.command * DT
        pos = pos + vel * DT

        history[step] = pos
        pairs = [
            float(np.linalg.norm(pos[i] - pos[j]))
            for i in range(N_AGENTS)
            for j in range(i + 1, N_AGENTS)
        ]
        closest[step] = min(pairs)
        furthest[step] = max(pairs)

    return history, closest, furthest, infeasible, range_hist


def main() -> None:
    ok, close_ok, far_ok, infeas_ok, live_range = _fly(SAFE_DISTANCE, COMM_RANGE)
    # Ask for more separation than the radio can span: no input satisfies both.
    _, close_bad, far_bad, infeas_bad, _ = _fly(COMM_RANGE * 1.6, COMM_RANGE)
    times = np.arange(STEPS) * DT

    logger = SimLogger("connectivity_annulus", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Composed CBF: separation and connectivity")
    logger.log_metadata("safe_distance_m", SAFE_DISTANCE)
    logger.log_metadata("comm_range_m", COMM_RANGE)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            positions=ok[step],
            closest_pair=float(close_ok[step]),
            furthest_pair=float(far_ok[step]),
            infeasible=bool(infeas_ok[step]),
        )
    logger.log_summary("min_closest_pair_m", float(close_ok.min()))
    logger.log_summary("max_furthest_pair_m", float(far_ok.max()))
    logger.log_summary("safe_distance_m", SAFE_DISTANCE)
    logger.log_summary("comm_range_m", COMM_RANGE)
    logger.log_summary("infeasible_steps", int(infeas_ok.sum()))
    logger.log_summary("infeasible_steps_impossible_case", int(infeas_bad.sum()))
    logger.save()

    skip = max(1, STEPS // 140)
    idx = list(range(0, STEPS, skip))
    colours = plt.cm.cool(np.linspace(0, 1, N_AGENTS))
    c_rgb = [c[:3] for c in colours]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pair = fig.add_subplot(gs[0, 1])
    ax_inf = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Connectivity Annulus — separation and connectivity enforced together", fontsize=13
    )

    ax3d.set_xlim(0, WORLD)
    ax3d.set_ylim(0, WORLD)
    ax3d.set_zlim(0, WORLD)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")

    ax_pair.set_xlim(0, times[-1])
    ax_pair.set_ylim(0, RANGE_MAX * 1.25)
    ax_pair.set_xlabel("Time [s]", fontsize=8)
    ax_pair.set_ylabel("Pair distance [m]", fontsize=8)
    ax_pair.set_title("The shell breathes with the radio", fontsize=9)
    ax_pair.grid(True, alpha=0.3)
    ax_pair.axhline(SAFE_DISTANCE, color="crimson", ls="--", lw=1.1, label="safe distance")
    (l_range,) = ax_pair.plot([], [], color="tab:blue", ls="--", lw=1.2, label="comm range")
    (l_close,) = ax_pair.plot([], [], color="tab:red", lw=1.5, label="closest pair")
    (l_far,) = ax_pair.plot([], [], color="tab:blue", lw=1.5, label="furthest pair")
    ax_pair.legend(fontsize=7, loc="center right")

    ax_inf.set_xlim(0, times[-1])
    ax_inf.set_ylim(-0.1, 1.2)
    ax_inf.set_xlabel("Time [s]", fontsize=8)
    ax_inf.set_ylabel("QP infeasible", fontsize=8)
    ax_inf.set_title("Ask the impossible and the filter says so", fontsize=9)
    ax_inf.grid(True, alpha=0.3)
    (i_ok,) = ax_inf.plot([], [], color="tab:green", lw=1.5, label=f"d={SAFE_DISTANCE:.0f} m")
    (i_bad,) = ax_inf.plot(
        [], [], color="tab:red", lw=1.5, label=f"d={COMM_RANGE * 1.6:.0f} m > comm range"
    )
    ax_inf.legend(fontsize=7, loc="center right")

    veh: list = []
    links: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("connectivity_annulus", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        p = ok[step]

        clear_vehicle_artists(veh)
        for artist in links:
            artist.remove()
        links.clear()

        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                (ln,) = ax3d.plot(
                    [p[i, 0], p[j, 0]],
                    [p[i, 1], p[j, 1]],
                    [p[i, 2], p[j, 2]],
                    color="0.6",
                    lw=0.5,
                    alpha=0.4,
                )
                links.append(ln)
            v = (ok[step, i] - ok[step - 1, i]) / DT if step > 0 else np.zeros(3)
            veh.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[i],
                    attitude_from_velocity(v),
                    size=1.8,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                    center_color=c_rgb[i],
                    motor_color=c_rgb[i],
                )
            )

        l_range.set_data(times[:step], live_range[:step])
        l_close.set_data(times[:step], close_ok[:step])
        l_far.set_data(times[:step], far_ok[:step])
        i_ok.set_data(times[:step], infeas_ok[:step])
        i_bad.set_data(times[:step], infeas_bad[:step])
        title.set_text(
            f"t = {step * DT:.1f} s   range {live_range[step]:.0f} m   "
            f"closest {close_ok[step]:.1f}   furthest {far_ok[step]:.1f}"
        )

    anim.animate(update, len(idx))
    anim.save()

    print(f"  feasible case  closest {close_ok.min():.2f} m  furthest {far_ok.max():.2f} m")
    print(f"                 limits: separation {SAFE_DISTANCE} m, comm range {COMM_RANGE} m")
    print(f"  impossible case infeasible on {int(infeas_bad.sum())}/{STEPS} steps")


if __name__ == "__main__":
    main()
