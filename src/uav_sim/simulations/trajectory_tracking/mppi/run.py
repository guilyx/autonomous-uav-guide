# Erwin Lejeune - 2026-02-19
"""MPPI as an online local planner tracking a global reference path.

A global path is generated via A* (or provided), then MPPI runs as a
*local planner* at 50 Hz on a point-mass model. At each step it picks
a look-ahead sub-goal from the global path and optimises a short
horizon trajectory towards it. The resulting velocity command is sent
to the cascaded PID controller.

The drone takes off, then follows the path all the way to the goal.

Reference: G. Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning," ICRA, 2017. DOI: 10.1109/ICRA.2017.7989202
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.path_tracking.flight_ops import takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_tracking.mppi import MPPITracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
DT_SIM = 0.005
DT_MPPI = 0.02  # 50 Hz local planning

_OBS_SPHERES: list[tuple[np.ndarray, float]] = []


def _box_to_sphere(b: BoxObstacle) -> tuple[np.ndarray, float]:
    centre = (b.min_corner + b.max_corner) / 2
    half_xy = (b.max_corner[:2] - b.min_corner[:2]) / 2
    radius = float(np.linalg.norm(half_xy)) * 1.1
    return (centre, radius)


def _dyn(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    pos, vel = x[:3], x[3:6]
    nv = vel + u * dt
    np_ = pos + nv * dt
    return np.concatenate([np_, nv])


def _cost(x: np.ndarray, u: np.ndarray, ref: np.ndarray | None) -> float:
    if ref is None:
        return 0.0
    goal_cost = float(np.sum((x[:3] - ref[:3]) ** 2))
    vel_cost = 0.05 * float(np.sum(u**2))
    obs_cost = 0.0
    for c, r in _OBS_SPHERES:
        dist = float(np.linalg.norm(x[:3] - c))
        if dist < r + 1.5:
            obs_cost += 5e3 * max(0.0, r + 1.5 - dist) ** 2
    return goal_cost + vel_cost + obs_cost


def main() -> None:
    global _OBS_SPHERES  # noqa: PLW0603
    world, buildings = default_world()
    _OBS_SPHERES = [_box_to_sphere(b) for b in buildings]

    global_path = np.array(
        [
            [3.0, 3.0, CRUISE_ALT],
            [10.0, 5.0, CRUISE_ALT],
            [15.0, 15.0, CRUISE_ALT],
            [20.0, 25.0, CRUISE_ALT],
            [27.0, 27.0, CRUISE_ALT],
        ]
    )

    tracker = MPPITracker(
        state_dim=6,
        control_dim=3,
        horizon=15,
        num_samples=200,
        lambda_=0.5,
        control_std=np.array([1.0, 1.0, 0.5]),
        dynamics=_dyn,
        cost_fn=_cost,
        dt=DT_MPPI,
    )

    pursuit = PurePursuit3D(lookahead=5.0, waypoint_threshold=2.0, adaptive=True)

    quad = Quadrotor()
    quad.reset(position=np.array([global_path[0, 0], global_path[0, 1], 0.0]))
    ctrl = CascadedPIDController()

    states_list: list[np.ndarray] = []
    local_goals: list[np.ndarray] = []
    rollout_snapshots: list[np.ndarray | None] = []

    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=DT_SIM, duration=3.0, states=states_list)

    sim_steps_per_mppi = max(1, int(DT_MPPI / DT_SIM))
    max_mppi_steps = 3000
    seed_counter = 0

    for step_i in range(max_mppi_steps):
        s = quad.state
        if not (np.all(np.isfinite(s[:3])) and np.all(np.abs(s[:3]) < 500)):
            break

        vel = s[6:9] if len(s) >= 9 else None
        local_goal = pursuit.compute_target(s[:3], global_path, velocity=vel)
        local_goals.append(local_goal.copy())

        mppi_state = np.concatenate([s[:3], s[6:9]])
        mppi_ref = np.concatenate([local_goal, np.zeros(3)])
        result = tracker.compute(
            mppi_state, reference=mppi_ref, seed=seed_counter, return_rollouts=True
        )
        u_mppi, rollouts = result
        seed_counter += 1
        rollout_snapshots.append(rollouts.copy())

        desired_pos = s[:3] + u_mppi * DT_MPPI * 2.0
        desired_pos = np.clip(desired_pos, 0.0, WORLD_SIZE)

        for _ in range(sim_steps_per_mppi):
            states_list.append(quad.state.copy())
            wrench = ctrl.compute(quad.state, desired_pos, dt=DT_SIM)
            quad.step(wrench, DT_SIM)

        if pursuit.is_path_complete(s[:3], global_path):
            break

    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    flight_pos = flight_states[:, :3]
    n_total = len(flight_pos)

    # ── Animation ─────────────────────────────────────────────────────
    skip = max(1, n_total // 200)
    frames = list(range(0, n_total, skip))
    n_frames = len(frames)

    viz = ThreePanelViz(title="MPPI Local Planner — Online Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(global_path, color="cyan", lw=1.5, alpha=0.4, label="Global Path")
    viz.mark_start_goal(global_path[0], global_path[-1])

    trail_arts = viz.create_trail_artists(color="orange")
    (lg_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10, label="Local Goal")
    (lg_top,) = viz.ax_top.plot([], [], "r*", ms=8, zorder=10)

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

    anim = SimAnimator("mppi", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    local_goal_arr = np.array(local_goals) if local_goals else np.zeros((1, 3))

    def update(f: int) -> None:
        k = frames[f]
        viz.update_trail(trail_arts, flight_pos, k)
        viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)

        lg_idx = min(k // sim_steps_per_mppi, len(local_goal_arr) - 1)
        lg = local_goal_arr[lg_idx]
        lg_3d.set_data([lg[0]], [lg[1]])
        lg_3d.set_3d_properties([lg[2]])
        lg_top.set_data([lg[0]], [lg[1]])

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

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
