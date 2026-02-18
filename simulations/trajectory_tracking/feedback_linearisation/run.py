# Erwin Lejeune - 2026-02-15
"""Feedback-linearisation tracker: 3-panel circular tracking.

The quadrotor tracks a slow circular reference trajectory using
differential-flatness-based feedback linearisation, rendered in an
urban 30x30x30 world.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011, Sec. IV. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.trajectory_tracking.feedback_linearisation import (
    FeedbackLinearisationTracker,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CENTER = np.array([15.0, 15.0])
RADIUS = 6.0
ALT = 12.0
OMEGA = 0.25  # slow angular velocity [rad/s]


def _ref(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Circular reference: position, velocity, acceleration."""
    rp = np.array(
        [
            CENTER[0] + RADIUS * np.cos(OMEGA * t),
            CENTER[1] + RADIUS * np.sin(OMEGA * t),
            ALT + 1.5 * np.sin(0.3 * t),
        ]
    )
    rv = np.array(
        [
            -RADIUS * OMEGA * np.sin(OMEGA * t),
            RADIUS * OMEGA * np.cos(OMEGA * t),
            1.5 * 0.3 * np.cos(0.3 * t),
        ]
    )
    ra = np.array(
        [
            -RADIUS * OMEGA**2 * np.cos(OMEGA * t),
            -RADIUS * OMEGA**2 * np.sin(OMEGA * t),
            -1.5 * 0.3**2 * np.sin(0.3 * t),
        ]
    )
    return rp, rv, ra


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=4, seed=5)

    quad = Quadrotor()
    rp0, _, _ = _ref(0.0)
    quad.reset(position=rp0.copy())
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))

    tracker = FeedbackLinearisationTracker(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )

    dt, dur = 0.005, 25.0
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    refs = np.zeros((steps, 3))
    for i in range(steps):
        t = i * dt
        rp, rv, ra = _ref(t)
        refs[i] = rp
        states[i] = quad.state
        quad.step(tracker.compute(quad.state, rp, rv, ra), dt)

    pos = states[:, :3]

    # ── visualisation ──────────────────────────────────────────────────
    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    viz = ThreePanelViz(title="Feedback Linearisation — Circular Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(refs, color="red", lw=1.0, alpha=0.3, label="Reference")

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
