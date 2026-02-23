# Erwin Lejeune - 2026-02-19
"""MPPI trajectory tracking — figure-8 with sampled trajectories.

Tracks a figure-8 reference using MPPI as a local planner. All sampled
rollouts are rendered (faded cyan) with the chosen optimal trajectory
highlighted in green. A data panel shows tracking error and speed.

Reference: G. Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning," ICRA, 2017. DOI: 10.1109/ICRA.2017.7989202
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.simulations.common import (
    WORLD_SIZE,
    figure_8_ref,
    frame_indices,
)
from uav_sim.trajectory_tracking.mppi import MPPITracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

DT_SIM = 0.005
DT_MPPI = 0.02

MASS = 1.5
GRAVITY = 9.81
HOVER_T = MASS * GRAVITY


def _dyn(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Simplified 6-state (pos, vel) dynamics driven by wrench u=[T, tx, ty, tz]."""
    pos, vel = x[:3].copy(), x[3:6].copy()
    acc = np.array([0.0, 0.0, u[0] / MASS - GRAVITY])
    acc[0] += u[2] * 2.0 / MASS
    acc[1] += -u[1] * 2.0 / MASS
    nv = vel + acc * dt
    np_ = pos + nv * dt
    return np.concatenate([np_, nv])


def _cost(x: np.ndarray, u: np.ndarray, ref: np.ndarray | None) -> float:
    if ref is None:
        return 0.0
    pos_cost = float(np.sum((x[:3] - ref[:3]) ** 2)) * 10.0
    vel_cost = float(np.sum((x[3:6] - ref[3:6]) ** 2))
    ctrl_cost = 0.001 * float(np.sum((u - np.array([HOVER_T, 0, 0, 0])) ** 2))
    return pos_cost + vel_cost + ctrl_cost


def main() -> None:
    world, buildings = default_world()

    tracker = MPPITracker(
        state_dim=6,
        control_dim=4,
        horizon=25,
        num_samples=256,
        lambda_=0.5,
        control_std=np.array([2.0, 0.3, 0.3, 0.3]),
        dynamics=_dyn,
        cost_fn=_cost,
        dt=DT_MPPI,
    )
    tracker.U[:, 0] = HOVER_T

    quad = Quadrotor()
    rp0, _ = figure_8_ref(0.0)
    quad.reset(position=rp0.copy())
    init_hover(quad)

    dur = 30.0
    sim_steps_per_mppi = max(1, int(DT_MPPI / DT_SIM))
    max_mppi_steps = int(dur / DT_MPPI)

    states_list: list[np.ndarray] = []
    refs_list: list[np.ndarray] = []
    rollout_snapshots: list[np.ndarray | None] = []
    seed_counter = 0

    for ci in range(max_mppi_steps):
        s = quad.state
        if not (np.all(np.isfinite(s[:3])) and np.all(np.abs(s[:3]) < 500)):
            break

        t = ci * DT_MPPI
        rp, rv = figure_8_ref(t)

        mppi_state = np.concatenate([s[:3], s[6:9]])
        mppi_ref = np.concatenate([rp, rv])
        result = tracker.compute(
            mppi_state, reference=mppi_ref, seed=seed_counter, return_rollouts=True
        )
        wrench_mppi, rollouts = result
        seed_counter += 1
        rollout_snapshots.append(rollouts.copy())

        for _ in range(sim_steps_per_mppi):
            states_list.append(quad.state.copy())
            refs_list.append(rp.copy())
            quad.step(wrench_mppi, DT_SIM)

    states = np.array(states_list) if states_list else np.zeros((1, 12))
    refs = np.array(refs_list) if refs_list else np.zeros((1, 3))
    pos = states[:, :3]
    n_total = len(pos)
    times = np.arange(n_total) * DT_SIM
    err = np.linalg.norm(pos - refs, axis=1)
    speed = np.linalg.norm(states[:, 6:9], axis=1)

    logger = SimLogger("mppi", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "MPPI")
    logger.log_metadata("dt_sim", DT_SIM)
    logger.log_metadata("dt_mppi", DT_MPPI)
    logger.log_metadata("duration", times[-1] if len(times) > 0 else 0.0)
    for i in range(n_total):
        logger.log_step(
            t=times[i],
            position=pos[i],
            velocity=states[i, 6:9],
            tracking_error=err[i],
            reference=refs[i],
        )
    logger.log_summary("mean_error_m", float(err.mean()))
    logger.log_summary("max_error_m", float(err.max()))
    logger.log_summary("mean_speed_mps", float(speed.mean()))
    logger.save()

    # ── Animation ─────────────────────────────────────────────────────
    idx = frame_indices(n_total, max_frames=60)
    n_frames = len(idx)

    viz = ThreePanelViz(title="MPPI — Figure-8 Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(refs, color="red", lw=1.0, alpha=0.3, label="Reference")

    trail_arts = viz.create_trail_artists(color="orange")
    (ref_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10)
    (ref_top,) = viz.ax_top.plot([], [], "r*", ms=8, zorder=10)

    n_show = 30
    rollout_lines_3d = []
    rollout_lines_top = []
    for _ in range(n_show):
        (ln3d,) = viz.ax3d.plot([], [], [], "c-", lw=0.3, alpha=0.15)
        rollout_lines_3d.append(ln3d)
        (lnt,) = viz.ax_top.plot([], [], "c-", lw=0.3, alpha=0.15)
        rollout_lines_top.append(lnt)
    (opt_3d,) = viz.ax3d.plot([], [], [], "lime", lw=1.5, alpha=0.8, label="MPPI Optimal")
    (opt_top,) = viz.ax_top.plot([], [], "lime", lw=1.2, alpha=0.7)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    ax_d = viz.setup_data_axes(title="Tracking Error [m]", ylabel="Error")
    ax_d.set_xlim(0, dur)
    ax_d.set_ylim(0, max(0.5, err.max() * 1.1))
    (l_err,) = ax_d.plot([], [], "r-", lw=0.8, label="||e||")
    ax_v = ax_d.twinx()
    ax_v.set_ylabel("Speed [m/s]", fontsize=7)
    ax_v.tick_params(labelsize=6)
    ax_v.set_ylim(0, max(1.0, speed.max() * 1.2))
    (l_spd,) = ax_v.plot([], [], "b-", lw=0.5, alpha=0.6, label="speed")
    ax_d.legend(fontsize=5, loc="upper right")
    ax_v.legend(fontsize=5, loc="lower right")

    anim = SimAnimator("mppi", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig
    title = viz.ax3d.set_title("MPPI")

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail_arts, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

        ref_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_3d.set_3d_properties([refs[k, 2]])
        ref_top.set_data([refs[k, 0]], [refs[k, 1]])

        mppi_idx = min(k // sim_steps_per_mppi, len(rollout_snapshots) - 1)
        rolls = rollout_snapshots[mppi_idx]
        if rolls is not None:
            rng_vis = np.random.default_rng(f)
            sample_ids = rng_vis.choice(len(rolls), size=min(n_show, len(rolls)), replace=False)
            for j, ln3d in enumerate(rollout_lines_3d):
                if j < len(sample_ids):
                    r = rolls[sample_ids[j]]
                    ln3d.set_data(r[:, 0], r[:, 1])
                    ln3d.set_3d_properties(r[:, 2])
                    rollout_lines_top[j].set_data(r[:, 0], r[:, 1])
                else:
                    ln3d.set_data([], [])
                    ln3d.set_3d_properties([])
                    rollout_lines_top[j].set_data([], [])

            mean_roll = np.mean(rolls, axis=0)
            opt_3d.set_data(mean_roll[:, 0], mean_roll[:, 1])
            opt_3d.set_3d_properties(mean_roll[:, 2])
            opt_top.set_data(mean_roll[:, 0], mean_roll[:, 1])

        l_err.set_data(times[:k], err[:k])
        l_spd.set_data(times[:k], speed[:k])
        title.set_text(f"MPPI — t={times[k]:.1f}s  err={err[k]:.2f}m")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
