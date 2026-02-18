# Erwin Lejeune - 2026-02-15
"""NMPC tracking a PRM-planned path: 3-panel, two-phase visualisation.

Phase 1 — PRM builds a roadmap and finds a collision-free path.
Phase 2 — A nonlinear MPC tracker follows the smoothed PRM path,
           applying receding-horizon optimal control at every step.

References:
  [1] L. E. Kavraki et al., "Probabilistic Roadmaps," IEEE T-RA, 1996.
  [2] M. Diehl et al., "Real-Time Optimization and NMPC," J. Process
      Control, 2002.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_planning.prm_3d import PRM3D
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_tracking.nmpc import NMPCTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def _box_to_sphere(b):
    """Use XY-only bounding radius so planner can fly above buildings."""
    centre = (b.min_corner + b.max_corner) / 2
    half_xy = (b.max_corner[:2] - b.min_corner[:2]) / 2
    radius = float(np.linalg.norm(half_xy)) * 1.1
    return (centre, radius)


def main() -> None:
    world, buildings = default_world()
    sphere_obs = [_box_to_sphere(b) for b in buildings]

    start = np.array([3.0, 3.0, CRUISE_ALT])
    goal = np.array([27.0, 27.0, CRUISE_ALT])

    # Phase 1: PRM planning
    prm = PRM3D(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
        obstacles=sphere_obs,
        n_samples=300,
        k_neighbours=12,
    )
    prm.build(seed=42)
    path = prm.plan(start, goal)
    if path is None:
        print("PRM failed!")
        return
    raw_path = np.array(path)
    smooth = smooth_path_3d(raw_path, epsilon=2.0, min_spacing=1.5)

    # Phase 2: NMPC tracking with pure-pursuit for carrot selection
    quad = Quadrotor()
    quad.reset(position=start.copy())

    nmpc = NMPCTracker(
        horizon=8,
        dt=0.02,
        mass=quad.params.mass,
        gravity=quad.params.gravity,
        inertia=quad.params.inertia,
    )
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    dt, timeout = 0.005, 60.0
    ctrl_dt = 0.02
    max_steps = int(timeout / dt)
    states_list: list[np.ndarray] = []
    wrench = quad.hover_wrench()
    ctrl_counter = 0.0

    for _ in range(max_steps):
        s = quad.state
        if np.any(np.isnan(s[:3])) or np.any(np.abs(s[:3]) > 500):
            break
        states_list.append(s.copy())
        if pursuit.is_path_complete(s[:3], smooth):
            break

        ctrl_counter += dt
        if ctrl_counter >= ctrl_dt - 1e-8:
            vel = s[6:9] if len(s) >= 9 else None
            target = pursuit.compute_target(s[:3], smooth, velocity=vel)
            direction = target - s[:3]
            dist = np.linalg.norm(direction)
            ref_vel = direction / max(dist, 0.01) * 1.5 if dist > 0.1 else np.zeros(3)
            wrench = nmpc.compute(s, target, ref_vel=ref_vel)
            ctrl_counter = 0.0
        quad.step(wrench, dt)

    states = np.array(states_list) if states_list else np.zeros((1, 12))
    flight_pos = states[:, :3]

    # ── Animation ─────────────────────────────────────────────────────
    roadmap_pause = 20
    fly_step = max(1, len(flight_pos) // 120)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_ff = len(fly_frames)
    total = roadmap_pause + n_ff

    viz = ThreePanelViz(title="NMPC + PRM Path Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(start, goal)

    # Roadmap edges (thin lines)
    for i, adj in enumerate(prm.edges):
        for j, _ in adj:
            if j > i:
                p1, p2 = prm.nodes[i], prm.nodes[j]
                viz.ax3d.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    "c-",
                    lw=0.15,
                    alpha=0.0,
                )
    edge_lines = viz.ax3d.lines[:]

    viz.draw_path(smooth, color="lime", lw=2.0, alpha=0.0, label="Smoothed PRM")

    fly_trail = viz.create_trail_artists(color="orange")
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Phase 1: PRM Planning")

    # Get the smoothed path line artists (last added by draw_path)
    smooth_lines = [viz.ax3d.lines[-1], viz.ax_top.lines[-1], viz.ax_side.lines[-1]]

    anim = SimAnimator("nmpc_prm", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < roadmap_pause:
            frac = min(1.0, (f + 1) / (roadmap_pause * 0.5))
            for ln in edge_lines:
                ln.set_alpha(frac * 0.1)
            if f >= roadmap_pause // 2:
                for ln in smooth_lines:
                    ln.set_alpha(1.0)
                for ln in edge_lines:
                    ln.set_alpha(0.05)
            pct = int(100 * (f + 1) / roadmap_pause)
            title.set_text(f"Phase 1: PRM + Smooth — {pct}%")
        else:
            fi = f - roadmap_pause
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], states[k, 3:6], size=1.5)
            title.set_text("Phase 2: NMPC Tracking PRM Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
