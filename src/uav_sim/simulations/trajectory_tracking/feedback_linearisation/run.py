# Erwin Lejeune - 2026-02-15
"""Feedback-linearisation tracker: 3-panel path tracking to goal.

The quadrotor tracks a planned reference trajectory from start to goal
using differential-flatness-based feedback linearisation, navigating
through an urban 30x30x30 world with obstacle avoidance.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011, Sec. IV. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline

from uav_sim.environment import default_world
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles
from uav_sim.trajectory_tracking.feedback_linearisation import (
    FeedbackLinearisationTracker,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
START = np.array([3.0, 3.0, 12.0])
GOAL = np.array([27.0, 27.0, 12.0])
FLIGHT_SPEED = 2.0


def _time_parametrize(
    path: np.ndarray, speed: float
) -> tuple[np.ndarray, CubicSpline, CubicSpline, CubicSpline]:
    """Build time-parametrised cubic splines for position, velocity, acceleration."""
    dists = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
    times = dists / speed
    cs_x = CubicSpline(times, path[:, 0])
    cs_y = CubicSpline(times, path[:, 1])
    cs_z = CubicSpline(times, path[:, 2])
    return times, cs_x, cs_y, cs_z


def main() -> None:
    world, buildings = default_world()

    planned = plan_through_obstacles(buildings, START, GOAL, world_size=int(WORLD_SIZE))
    if planned is None:
        print("No path found!")
        return

    t_arr, cs_x, cs_y, cs_z = _time_parametrize(planned, FLIGHT_SPEED)
    t_final = t_arr[-1]

    quad = Quadrotor()
    quad.reset(position=START.copy())
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))

    tracker = FeedbackLinearisationTracker(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )

    dt = 0.005
    dur = t_final + 2.0
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    refs = np.zeros((steps, 3))
    for i in range(steps):
        t = min(i * dt, t_final)
        rp = np.array([float(cs_x(t)), float(cs_y(t)), float(cs_z(t))])
        rv = np.array([float(cs_x(t, 1)), float(cs_y(t, 1)), float(cs_z(t, 1))])
        ra = np.array([float(cs_x(t, 2)), float(cs_y(t, 2)), float(cs_z(t, 2))])
        if i * dt > t_final:
            rp = GOAL.copy()
            rv = np.zeros(3)
            ra = np.zeros(3)
        refs[i] = rp
        states[i] = quad.state
        quad.step(tracker.compute(quad.state, rp, rv, ra), dt)

    pos = states[:, :3]

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(
        title="Feedback Linearisation — Path Tracking to Goal",
        world_size=WORLD_SIZE,
    )
    viz.draw_buildings(buildings)
    viz.draw_path(planned, color="blue", lw=1.0, alpha=0.3, label="Planned Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()
    (ref_dot_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10)
    (ref_dot_top,) = viz.ax_top.plot([], [], "r*", ms=8)
    (ref_dot_side,) = viz.ax_side.plot([], [], "r*", ms=8)

    anim = SimAnimator("feedback_linearisation", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        ref_dot_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_dot_3d.set_3d_properties([refs[k, 2]])
        ref_dot_top.set_data([refs[k, 0]], [refs[k, 1]])
        ref_dot_side.set_data([refs[k, 0]], [refs[k, 2]])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
