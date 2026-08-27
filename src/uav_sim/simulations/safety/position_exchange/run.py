# Erwin Lejeune - 2026-08-27
"""Position exchange: eight agents swap across a ring under a CBF filter.

Every agent is commanded straight at the point diametrically opposite it,
so every straight-line path crosses the centre at the same moment. The
nominal controller is a plain PD that knows nothing about the others; a
control barrier function filter sits between it and the vehicles and edits
the command only when a pair is about to close inside the safe distance.

Three runs, same nominal controller, so the two separate claims can be told
apart:

* **unfiltered** -- the ring passes through itself, closing to centimetres.
* **filtered** -- nothing ever violates the barrier, and the fleet
  *deadlocks*. Eight agents pressed symmetrically against each other have
  no direction left that the QP will allow, and the swap never finishes.
* **filtered + swirl** -- a small tangential bias, applied only while the
  fleet is jammed, breaks the symmetry and the swap completes.

That middle case is the point of the demo. A CBF guarantees *safety* and
says nothing about *liveness*: the QP is perfectly happy to hold a fleet
still forever, because standing still is safe. Getting anywhere is the
nominal controller's job, and under perfect symmetry it needs help.

Reference: U. Borrmann et al., "Control Barrier Certificates for Safe Swarm
Behavior," IFAC ADHS, 2015. DOI: 10.1016/j.ifacol.2015.11.154
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.logging import SimLogger
from uav_sim.safety import SafeDistanceBarrier, SafetyFilter, SpeedLimitBarrier
from uav_sim.visualization import SimAnimator, attitude_from_velocity
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

WORLD_SIZE = 40.0
CENTRE = WORLD_SIZE / 2.0
CRUISE_ALT = 20.0
N_AGENTS = 8
RING_RADIUS = 12.0
SAFE_DISTANCE = 1.8
MAX_SPEED = 6.0
MAX_ACCEL = 8.0
SWIRL_GAIN = 4.0
DT = 0.02
STEPS = 3000


def _ring() -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, N_AGENTS, endpoint=False)
    start = np.column_stack(
        [
            RING_RADIUS * np.cos(angles) + CENTRE,
            RING_RADIUS * np.sin(angles) + CENTRE,
            np.full(N_AGENTS, CRUISE_ALT),
        ]
    )
    return start, 2.0 * np.array([CENTRE, CENTRE, CRUISE_ALT]) - start


def _swirl(positions: np.ndarray, closest: float) -> np.ndarray:
    """Tangential bias, faded in only while the fleet is jammed.

    A constant swirl would drag every agent off its goal for the whole run;
    gating it on the closest pair means it is zero in open space and only
    appears once the QP has the fleet pinned against the barrier.
    """
    radial = positions - np.array([CENTRE, CENTRE, CRUISE_ALT])
    radial[:, 2] = 0.0
    radius = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    tangent = np.column_stack([-radial[:, 1], radial[:, 0], np.zeros(len(positions))]) / radius
    jam = float(np.clip((SAFE_DISTANCE * 2.2 - closest) / (SAFE_DISTANCE * 1.2), 0.0, 1.0))
    return SWIRL_GAIN * jam * tangent


def _closest_pair(positions: np.ndarray) -> float:
    return min(
        float(np.linalg.norm(positions[i] - positions[j]))
        for i in range(N_AGENTS)
        for j in range(i + 1, N_AGENTS)
    )


def _fly(use_filter: bool, swirl: bool):
    """Run the exchange once, returning its trace."""
    pos, goal = _ring()
    vel = np.zeros_like(pos)
    filt = SafetyFilter(
        [SafeDistanceBarrier(SAFE_DISTANCE, k1=6.0, k2=6.0), SpeedLimitBarrier(MAX_SPEED)],
        u_min=-MAX_ACCEL,
        u_max=MAX_ACCEL,
    )

    history = np.zeros((STEPS, N_AGENTS, 3))
    separation = np.zeros(STEPS)
    goal_error = np.zeros(STEPS)
    correction = np.zeros(STEPS)

    for step in range(STEPS):
        closest = _closest_pair(pos)
        nominal = 3.0 * (goal - pos) - 3.5 * vel
        if swirl:
            nominal = nominal + _swirl(pos, closest)
        nominal = np.clip(nominal, -MAX_ACCEL, MAX_ACCEL)

        if use_filter:
            report = filt(pos, vel, nominal)
            command = report.command
            correction[step] = report.correction_norm
        else:
            command = nominal

        vel = vel + command * DT
        pos = pos + vel * DT

        history[step] = pos
        separation[step] = closest
        goal_error[step] = float(np.mean(np.linalg.norm(pos - goal, axis=1)))

    return history, separation, goal_error, correction, goal


def main() -> None:
    live, sep_live, err_live, correction, goal = _fly(use_filter=True, swirl=True)
    _, sep_lock, err_lock, _, _ = _fly(use_filter=True, swirl=False)
    _, sep_off, err_off, _, _ = _fly(use_filter=False, swirl=False)

    times = np.arange(STEPS) * DT

    logger = SimLogger("position_exchange", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "CBF-QP safety filter")
    logger.log_metadata("n_agents", N_AGENTS)
    logger.log_metadata("safe_distance_m", SAFE_DISTANCE)
    logger.log_metadata("dt", DT)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            positions=live[step],
            min_separation=float(sep_live[step]),
            min_separation_deadlocked=float(sep_lock[step]),
            min_separation_unfiltered=float(sep_off[step]),
            goal_error=float(err_live[step]),
            filter_correction=float(correction[step]),
        )
    logger.log_summary("safe_distance_m", SAFE_DISTANCE)
    logger.log_summary("min_separation_m", float(sep_live.min()))
    logger.log_summary("min_separation_unfiltered_m", float(sep_off.min()))
    logger.log_summary("barrier_held", bool(sep_live.min() >= SAFE_DISTANCE * 0.98))
    logger.log_summary("final_goal_error_m", float(err_live[-1]))
    logger.log_summary("final_goal_error_no_swirl_m", float(err_lock[-1]))
    logger.save()

    skip = max(1, STEPS // 150)
    idx = list(range(0, STEPS, skip))
    colours = plt.cm.tab10(np.linspace(0, 1, N_AGENTS))
    c_rgb = [c[:3] for c in colours]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_sep = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Position Exchange — a CBF filter around a PD controller that cannot see the others",
        fontsize=13,
    )

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.scatter(goal[:, 0], goal[:, 1], goal[:, 2], c=colours, s=45, marker="x", alpha=0.6)

    ax_sep.set_xlim(0, times[-1])
    ax_sep.set_ylim(0, 12)
    ax_sep.set_xlabel("Time [s]", fontsize=8)
    ax_sep.set_ylabel("Closest pair [m]", fontsize=8)
    ax_sep.set_title("Separation — safety is the barrier's only claim", fontsize=9)
    ax_sep.grid(True, alpha=0.3)
    ax_sep.axhline(SAFE_DISTANCE, color="crimson", ls="--", lw=1.2, label="safe distance")
    (l_live,) = ax_sep.plot([], [], color="tab:green", lw=1.5, label="filtered + swirl")
    (l_lock,) = ax_sep.plot([], [], color="tab:blue", lw=1.1, alpha=0.8, label="filtered")
    (l_off,) = ax_sep.plot([], [], color="tab:red", lw=1.1, alpha=0.8, label="unfiltered")
    ax_sep.legend(fontsize=7, loc="upper right")

    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(0, max(err_live.max(), err_off.max()) * 1.1)
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Mean distance to goal [m]", fontsize=8)
    ax_err.set_title("Liveness is not — the symmetric fleet deadlocks", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    (e_live,) = ax_err.plot([], [], color="tab:green", lw=1.5, label="filtered + swirl")
    (e_lock,) = ax_err.plot([], [], color="tab:blue", lw=1.1, alpha=0.8, label="filtered")
    (e_off,) = ax_err.plot([], [], color="tab:red", lw=1.1, alpha=0.8, label="unfiltered")
    ax_err.legend(fontsize=7, loc="upper right")

    trails = [
        ax3d.plot([], [], [], color=c_rgb[i], lw=0.9, alpha=0.55)[0] for i in range(N_AGENTS)
    ]
    veh: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("position_exchange", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        p = live[step]

        clear_vehicle_artists(veh)
        for i in range(N_AGENTS):
            vel = (live[step, i] - live[step - 1, i]) / DT if step > 0 else np.zeros(3)
            veh.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[i],
                    attitude_from_velocity(vel),
                    size=1.0,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                    center_color=c_rgb[i],
                    motor_color=c_rgb[i],
                )
            )
            trail = live[: step + 1 : max(1, step // 60 or 1), i]
            trails[i].set_data(trail[:, 0], trail[:, 1])
            trails[i].set_3d_properties(trail[:, 2])

        l_live.set_data(times[:step], sep_live[:step])
        l_lock.set_data(times[:step], sep_lock[:step])
        l_off.set_data(times[:step], sep_off[:step])
        e_live.set_data(times[:step], err_live[:step])
        e_lock.set_data(times[:step], err_lock[:step])
        e_off.set_data(times[:step], err_off[:step])
        title.set_text(
            f"t = {step * DT:.1f} s   closest pair {sep_live[step]:.2f} m "
            f"(limit {SAFE_DISTANCE} m)"
        )

    anim.animate(update, len(idx))
    anim.save()

    for label, sep, err in (
        ("unfiltered", sep_off, err_off),
        ("filtered", sep_lock, err_lock),
        ("filtered + swirl", sep_live, err_live),
    ):
        print(f"  {label:<17} min separation {sep.min():.3f} m   goal error {err[-1]:.3f} m")
    print(f"  {'safe distance':<17} {SAFE_DISTANCE:.3f} m")


if __name__ == "__main__":
    main()
