# Erwin Lejeune - 2026-02-19
"""Frenet optimal trajectory as an iterative local planner.

A global reference path is given. At regular intervals the Frenet
planner re-plans a short horizon trajectory in the Frenet frame,
and the drone follows each local plan with pure-pursuit + PID.

Phase 1 — shows the initial candidate sampling & selection.
Phase 2 — online replanning: the drone follows successive Frenet
          arcs all the way to the goal.

Reference: M. Werling et al., "Optimal Trajectory Generation for Dynamic
Street Scenarios in a Frenet Frame," ICRA, 2010.
DOI: 10.1109/ROBOT.2010.5509799
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import fly_path, init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_planning.frenet_optimal import FrenetOptimalPlanner
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0


def main() -> None:
    world, buildings = default_world()
    sphere_obs = [
        (
            (b.min_corner + b.max_corner) / 2,
            float(np.linalg.norm((b.max_corner[:2] - b.min_corner[:2]) / 2)) * 1.1,
        )
        for b in buildings
    ]

    ref_path = np.array(
        [
            [3.0, 5.0, CRUISE_ALT],
            [8.0, 10.0, CRUISE_ALT],
            [15.0, 15.0, CRUISE_ALT],
            [22.0, 20.0, CRUISE_ALT],
            [27.0, 25.0, CRUISE_ALT],
        ]
    )

    planner = FrenetOptimalPlanner(
        max_lat_offset=3.0,
        n_lat_samples=9,
        min_speed=0.8,
        max_speed=3.0,
        n_speed_samples=5,
        dt=0.1,
        horizon=3.0,
        w_smooth=0.1,
        w_lateral=1.5,
        w_speed=0.5,
        target_speed=2.0,
    )

    # Show initial candidate sampling (for animation phase 1)
    best_init, candidates = planner.plan(
        ref_path, s0=0.0, d0=0.0, s_dot0=1.5, obstacles=sphere_obs
    )

    # Phase 2: iterative local replanning
    quad = Quadrotor()
    quad.reset(position=np.array([ref_path[0, 0], ref_path[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    states_list: list[np.ndarray] = []
    local_goals: list[np.ndarray] = []
    local_plans: list[np.ndarray] = []

    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=0.005, duration=3.0, states=states_list)
    local_goals.extend([ref_path[0]] * len(states_list))
    init_hover(quad)

    dists = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(ref_path, axis=0), axis=1)])
    total_dist = dists[-1]
    s_current = 0.0
    d_current = 0.0
    s_dot_current = 1.5
    max_replans = 40

    for _ in range(max_replans):
        if s_current >= total_dist - 1.0:
            break

        best, _ = planner.plan(
            ref_path,
            s0=s_current,
            d0=d_current,
            s_dot0=s_dot_current,
            obstacles=sphere_obs,
        )
        if best is None:
            break

        traj_pts = np.column_stack([best.x, best.y, best.z])
        local_plans.append(traj_pts.copy())

        n_before = len(states_list)
        fly_path(
            quad,
            ctrl,
            traj_pts,
            dt=0.005,
            pursuit=pursuit,
            timeout=8.0,
            states=states_list,
        )
        n_after = len(states_list)
        for i in range(n_before, n_after):
            lg = pursuit.compute_target(
                states_list[i][:3],
                traj_pts,
                velocity=states_list[i][6:9] if len(states_list[i]) >= 9 else None,
            )
            local_goals.append(lg)

        pos = quad.state[:3]
        # Estimate where we are on the reference path
        for j in range(len(dists) - 1):
            if dists[j + 1] >= s_current:
                seg_dir = ref_path[j + 1] - ref_path[j]
                seg_len = np.linalg.norm(seg_dir)
                if seg_len > 1e-6:
                    proj = float(np.dot(pos - ref_path[j], seg_dir / seg_len))
                    s_current = dists[j] + max(0.0, proj)
                break
        speed = float(np.linalg.norm(quad.state[6:9]))
        s_dot_current = max(0.5, speed)
        d_current = 0.0

        if float(np.linalg.norm(pos - ref_path[-1])) < 2.0:
            break

    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    flight_pos = flight_states[:, :3]
    n_total = len(flight_pos)

    # ── Animation ─────────────────────────────────────────────────────
    valid_cands = [c for c in candidates if c.cost < float("inf")]
    cand_pause = 25
    fly_skip = max(1, n_total // 150)
    fly_frames = list(range(0, n_total, fly_skip))
    n_ff = len(fly_frames)
    total_frames = cand_pause + n_ff

    viz = ThreePanelViz(
        title="Frenet Optimal — Iterative Local Planner",
        world_size=WORLD_SIZE,
    )
    viz.draw_buildings(buildings)
    viz.draw_path(ref_path, color="gray", lw=1.0, alpha=0.3, label="Global Reference")
    viz.mark_start_goal(ref_path[0], ref_path[-1])

    cand_lines_3d = []
    for c in valid_cands:
        (ln,) = viz.ax3d.plot(c.x, c.y, c.z, "c-", lw=0.4, alpha=0.0)
        cand_lines_3d.append(ln)

    (best_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Optimal")
    (best_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)
    (best_side,) = viz.ax_side.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists(color="orange")
    (lg_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10, label="Local Goal")
    (lg_top,) = viz.ax_top.plot([], [], "r*", ms=8, zorder=10)
    (arc_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.0, alpha=0.7, label="Local Arc")
    (arc_top,) = viz.ax_top.plot([], [], "lime", lw=1.5, alpha=0.6)
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title_art = viz.ax3d.set_title("Phase 1: Frenet Sampling")

    anim = SimAnimator("frenet_optimal", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    local_goal_arr = np.array(local_goals) if local_goals else np.zeros((1, 3))
    local_plan_arr = local_plans if local_plans else []

    def _plan_for_step(k: int) -> np.ndarray | None:
        """Find which local plan arc is active at step k."""
        pos_k = flight_pos[k]
        best_plan = None
        best_dist = float("inf")
        for plan in local_plan_arr:
            d = np.min(np.linalg.norm(plan - pos_k, axis=1))
            if d < best_dist:
                best_dist = d
                best_plan = plan
        return best_plan if best_dist < 5.0 else None

    def update(f: int) -> None:
        if f < cand_pause:
            frac = min(1.0, (f + 1) / (cand_pause * 0.4))
            for ln in cand_lines_3d:
                ln.set_alpha(frac * 0.25)
            if f >= cand_pause // 2 and best_init is not None:
                best_3d.set_alpha(1.0)
                best_3d.set_data(best_init.x, best_init.y)
                best_3d.set_3d_properties(best_init.z)
                best_top.set_alpha(1.0)
                best_top.set_data(best_init.x, best_init.y)
                best_side.set_alpha(1.0)
                best_side.set_data(best_init.x, best_init.z)
                for ln in cand_lines_3d:
                    ln.set_alpha(0.08)
            pct = int(100 * (f + 1) / cand_pause)
            title_art.set_text(f"Phase 1: Frenet — {len(valid_cands)} candidates — {pct}%")
        else:
            for ln in cand_lines_3d:
                ln.set_alpha(0.0)
            best_3d.set_alpha(0.0)
            best_top.set_alpha(0.0)
            best_side.set_alpha(0.0)

            fi = f - cand_pause
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)

            lg_idx = min(k, len(local_goal_arr) - 1)
            lg = local_goal_arr[lg_idx]
            lg_3d.set_data([lg[0]], [lg[1]])
            lg_3d.set_3d_properties([lg[2]])
            lg_top.set_data([lg[0]], [lg[1]])

            plan = _plan_for_step(k)
            if plan is not None:
                arc_3d.set_data(plan[:, 0], plan[:, 1])
                arc_3d.set_3d_properties(plan[:, 2])
                arc_top.set_data(plan[:, 0], plan[:, 1])
            else:
                arc_3d.set_data([], [])
                arc_3d.set_3d_properties([])
                arc_top.set_data([], [])

            if local_plan_arr:
                step_per_plan = max(1, n_total // len(local_plan_arr))
                replan_idx = min(len(local_plan_arr), k // step_per_plan + 1)
            else:
                replan_idx = 0
            n_plans = len(local_plan_arr)
            title_art.set_text(f"Phase 2: Frenet Replanning — arc {replan_idx}/{n_plans}")

    anim.animate(update, total_frames)
    anim.save()


if __name__ == "__main__":
    main()
