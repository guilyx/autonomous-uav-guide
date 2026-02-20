# Erwin Lejeune - 2026-02-19
"""Feedback-linearisation tracker: takeoff + full path tracking to goal.

The quadrotor takes off smoothly, then tracks a planned reference
trajectory from start to goal using differential-flatness-based
feedback linearisation, navigating through an urban 30x30x30 world.

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
from uav_sim.path_tracking.flight_ops import init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.trajectory_tracking.feedback_linearisation import (
    FeedbackLinearisationTracker,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
START = np.array([3.0, 3.0, CRUISE_ALT])
GOAL = np.array([27.0, 27.0, CRUISE_ALT])
FLIGHT_SPEED = 2.0


def _time_parametrize(
    path: np.ndarray, speed: float
) -> tuple[np.ndarray, CubicSpline, CubicSpline, CubicSpline]:
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
        planned = np.array([START, GOAL])

    t_arr, cs_x, cs_y, cs_z = _time_parametrize(planned, FLIGHT_SPEED)
    t_final = t_arr[-1]

    # Takeoff with PID first
    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))
    ctrl = CascadedPIDController()

    states_list: list[np.ndarray] = []
    refs_list: list[np.ndarray] = []
    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=0.005, duration=3.0, states=states_list)
    refs_list.extend([START.copy()] * len(states_list))

    init_hover(quad)

    tracker = FeedbackLinearisationTracker(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )

    dt = 0.005
    dur = t_final + 2.0
    steps = int(dur / dt)
    for i in range(steps):
        s = quad.state
        if not (np.all(np.isfinite(s[:3])) and np.all(np.abs(s[:3]) < 500)):
            break

        t = min(i * dt, t_final)
        rp = np.array([float(cs_x(t)), float(cs_y(t)), float(cs_z(t))])
        rv = np.array([float(cs_x(t, 1)), float(cs_y(t, 1)), float(cs_z(t, 1))])
        ra = np.array([float(cs_x(t, 2)), float(cs_y(t, 2)), float(cs_z(t, 2))])
        if i * dt > t_final:
            rp = GOAL.copy()
            rv = np.zeros(3)
            ra = np.zeros(3)

        refs_list.append(rp.copy())
        states_list.append(s.copy())
        quad.step(tracker.compute(s, rp, rv, ra), dt)

    states = np.array(states_list) if states_list else np.zeros((1, 12))
    refs = np.array(refs_list) if refs_list else np.zeros((1, 3))
    pos = states[:, :3]
    n_total = len(pos)

    # ── Visualisation ──────────────────────────────────────────────────
    skip = max(1, n_total // 200)
    idx = list(range(0, n_total, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(
        title="Feedback Linearisation — Path Tracking to Goal",
        world_size=WORLD_SIZE,
    )
    viz.draw_buildings(buildings)
    viz.draw_path(planned, color="blue", lw=1.0, alpha=0.3, label="Planned Path")
    viz.mark_start_goal(START, GOAL)

    trail = viz.create_trail_artists()
    (ref_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, label="Reference")
    (ref_top,) = viz.ax_top.plot([], [], "r*", ms=8)
    (ref_side,) = viz.ax_side.plot([], [], "r*", ms=8)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    anim = SimAnimator("feedback_linearisation", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)
        ref_3d.set_data([refs[k, 0]], [refs[k, 1]])
        ref_3d.set_3d_properties([refs[k, 2]])
        ref_top.set_data([refs[k, 0]], [refs[k, 1]])
        ref_side.set_data([refs[k, 0]], [refs[k, 2]])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
