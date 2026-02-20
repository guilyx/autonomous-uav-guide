# Erwin Lejeune - 2026-02-15
"""Frenet optimal trajectory: 3-panel visualisation.

Phase 1 — Algorithm: candidate Frenet trajectories are sampled along
           a reference path. The optimal (lowest-cost, collision-free)
           trajectory is highlighted.
Phase 2 — Platform: quadrotor follows the optimal trajectory using
           pure-pursuit + PID.

Reference: M. Werling et al., "Optimal Trajectory Generation for Dynamic
Street Scenarios in a Frenet Frame," ICRA, 2010.
DOI: 10.1109/ROBOT.2010.5509799
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_planning.frenet_optimal import FrenetOptimalPlanner
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


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
        horizon=5.0,
        w_smooth=0.1,
        w_lateral=1.5,
        w_speed=0.5,
        target_speed=2.0,
    )
    best, candidates = planner.plan(ref_path, s0=0.0, d0=0.0, s_dot0=1.5, obstacles=sphere_obs)
    if best is None:
        print("No feasible Frenet trajectory!")
        return

    traj_3d_pts = np.column_stack([best.x, best.y, best.z])

    # Phase 2: fly
    quad = Quadrotor()
    quad.reset(position=np.array([ref_path[0, 0], ref_path[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    flight_states = fly_mission(
        quad,
        ctrl,
        traj_3d_pts,
        cruise_alt=CRUISE_ALT,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.5,
        landing_duration=2.5,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation ─────────────────────────────────────────────────────
    valid_cands = [c for c in candidates if c.cost < float("inf")]
    cand_pause = 25
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_ff = len(fly_frames)
    total = cand_pause + n_ff

    viz = ThreePanelViz(
        title="Frenet Optimal Trajectory — Sample → Select → Flight", world_size=WORLD_SIZE
    )
    viz.draw_buildings(buildings)
    viz.draw_path(ref_path, color="gray", lw=1.0, alpha=0.3, label="Reference")
    viz.mark_start_goal(ref_path[0], ref_path[-1])

    # Pre-draw candidate paths (initially hidden)
    cand_lines_3d = []
    for c in valid_cands:
        (ln,) = viz.ax3d.plot(c.x, c.y, c.z, "c-", lw=0.4, alpha=0.0)
        cand_lines_3d.append(ln)

    (best_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Optimal")
    (best_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)
    (best_side,) = viz.ax_side.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Phase 1: Frenet Sampling")

    anim = SimAnimator("frenet_optimal", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < cand_pause:
            frac = min(1.0, (f + 1) / (cand_pause * 0.4))
            for ln in cand_lines_3d:
                ln.set_alpha(frac * 0.25)
            if f >= cand_pause // 2:
                best_3d.set_alpha(1.0)
                best_3d.set_data(best.x, best.y)
                best_3d.set_3d_properties(best.z)
                best_top.set_alpha(1.0)
                best_top.set_data(best.x, best.y)
                best_side.set_alpha(1.0)
                best_side.set_data(best.x, best.z)
                for ln in cand_lines_3d:
                    ln.set_alpha(0.08)
            pct = int(100 * (f + 1) / cand_pause)
            n_valid = len(valid_cands)
            title.set_text(f"Phase 1: Frenet — {n_valid} candidates — {pct}%")
        else:
            fi = f - cand_pause
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Following Frenet Optimal Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
