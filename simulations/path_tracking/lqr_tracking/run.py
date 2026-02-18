# Erwin Lejeune - 2026-02-15
"""LQR path tracking: 3-panel live simulation.

The quadrotor tracks a waypoint sequence using the LQR path tracker
which feeds time-varying references (position + heading velocity)
to the full-state LQR controller.

Reference: B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear
Quadratic Methods," Prentice-Hall, 1990.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_tracking.flight_ops import init_hover
from uav_sim.path_tracking.lqr_path_tracker import LQRPathTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, WORLD_SIZE))
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=5, seed=13)

    waypoints = np.array(
        [
            [4.0, 4.0, CRUISE_ALT],
            [12.0, 6.0, CRUISE_ALT + 2],
            [20.0, 14.0, CRUISE_ALT],
            [25.0, 22.0, CRUISE_ALT + 3],
            [20.0, 27.0, CRUISE_ALT],
        ]
    )

    quad = Quadrotor()
    quad.reset(position=waypoints[0].copy())
    init_hover(quad)

    tracker = LQRPathTracker(
        lookahead=3.0,
        speed=1.5,
        mass=quad.params.mass,
        gravity=quad.params.gravity,
        inertia=quad.params.inertia,
    )

    dt, timeout = 0.005, 50.0
    max_steps = int(timeout / dt)
    states_list: list[np.ndarray] = []
    for _ in range(max_steps):
        s = quad.state
        if np.any(np.isnan(s[:3])) or np.any(np.abs(s[:3]) > 500):
            break
        states_list.append(s.copy())
        if tracker.is_path_complete(s[:3], waypoints, threshold=1.5):
            break
        wrench = tracker.compute(s, waypoints)
        quad.step(wrench, dt)

    states = np.array(states_list) if states_list else np.zeros((1, 12))
    pos = states[:, :3]

    # ── Visualisation ─────────────────────────────────────────────────
    skip = max(1, len(states) // 200)
    idx = list(range(0, len(states), skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="LQR Path Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(waypoints, color="red", lw=1.0, alpha=0.5, label="Path")

    for i, wp in enumerate(waypoints):
        viz.ax3d.scatter(*wp, c="red", s=60, marker="D", zorder=5)
        viz.ax_top.plot(wp[0], wp[1], "rD", ms=5)
        viz.ax_side.plot(wp[0], wp[2], "rD", ms=5)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    trail = viz.create_trail_artists()
    anim = SimAnimator("lqr_tracking", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = idx[f]
        viz.update_trail(trail, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
