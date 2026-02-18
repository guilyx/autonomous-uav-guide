# Erwin Lejeune - 2026-02-15
"""Path smoothing demo: 3-panel visualisation of RDP + spline resampling.

Generates a noisy zigzag path (as would come from a grid-based planner),
then demonstrates the RDP simplification and cubic-spline resampling steps
side by side.  The drone then flies the smoothed path.

Reference: D. Douglas, T. Peucker, "Algorithms for the Reduction of the
Number of Points Required to Represent a Digitized Line or its Caricature,"
Cartographica, 1973.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import rdp_simplify, smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def _generate_zigzag(rng: np.random.Generator) -> np.ndarray:
    """Create a realistic noisy grid-planner path."""
    base = np.array(
        [
            [3.0, 3.0, CRUISE_ALT],
            [6.0, 4.0, CRUISE_ALT],
            [8.0, 7.0, CRUISE_ALT + 1],
            [10.0, 8.0, CRUISE_ALT],
            [12.0, 10.0, CRUISE_ALT + 2],
            [14.0, 13.0, CRUISE_ALT],
            [16.0, 14.0, CRUISE_ALT],
            [18.0, 16.0, CRUISE_ALT + 1],
            [20.0, 18.0, CRUISE_ALT],
            [22.0, 20.0, CRUISE_ALT + 2],
            [24.0, 22.0, CRUISE_ALT],
            [26.0, 25.0, CRUISE_ALT],
            [27.0, 27.0, CRUISE_ALT],
        ]
    )
    noise = rng.normal(0, 0.4, base.shape)
    noise[0] = 0
    noise[-1] = 0
    return base + noise


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, WORLD_SIZE),
    )
    buildings = add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=4, seed=77)

    rng = np.random.default_rng(42)
    raw_path = _generate_zigzag(rng)
    rdp_path = rdp_simplify(raw_path, epsilon=1.5)
    smooth_path = smooth_path_3d(raw_path, epsilon=1.5, min_spacing=1.0)

    # Fly the smoothed path
    quad = Quadrotor()
    quad.reset(position=np.array([raw_path[0, 0], raw_path[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    flight_states = fly_mission(
        quad,
        ctrl,
        smooth_path,
        cruise_alt=CRUISE_ALT,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.5,
        landing_duration=2.5,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation: 3 phases ───────────────────────────────────────────
    n_raw_show = 20
    n_rdp_show = 20
    n_smooth_show = 20
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_ff = len(fly_frames)
    total = n_raw_show + n_rdp_show + n_smooth_show + n_ff

    viz = ThreePanelViz(title="Path Smoothing Demo — RDP + Resample", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)

    # Pre-draw raw path (initially hidden)
    (raw_3d,) = viz.ax3d.plot([], [], [], "r-", lw=1.5, alpha=0.0, label="Raw (zigzag)")
    (raw_top,) = viz.ax_top.plot([], [], "r-", lw=1.5, alpha=0.0)
    (raw_side,) = viz.ax_side.plot([], [], "r-", lw=1.5, alpha=0.0)
    raw_scat_3d = viz.ax3d.scatter([], [], [], c="red", s=15, alpha=0.0)

    (rdp_3d,) = viz.ax3d.plot([], [], [], "b-", lw=2.0, alpha=0.0, label="RDP simplified")
    (rdp_top,) = viz.ax_top.plot([], [], "b-", lw=2.0, alpha=0.0)
    (rdp_side,) = viz.ax_side.plot([], [], "b-", lw=2.0, alpha=0.0)

    (sm_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Smoothed")
    (sm_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)
    (sm_side,) = viz.ax_side.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("")

    anim = SimAnimator("path_smoothing_demo", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        p1_end = n_raw_show
        p2_end = p1_end + n_rdp_show
        p3_end = p2_end + n_smooth_show

        if f < p1_end:
            raw_3d.set_alpha(1.0)
            raw_3d.set_data(raw_path[:, 0], raw_path[:, 1])
            raw_3d.set_3d_properties(raw_path[:, 2])
            raw_top.set_alpha(1.0)
            raw_top.set_data(raw_path[:, 0], raw_path[:, 1])
            raw_side.set_alpha(1.0)
            raw_side.set_data(raw_path[:, 0], raw_path[:, 2])
            raw_scat_3d._offsets3d = (raw_path[:, 0], raw_path[:, 1], raw_path[:, 2])
            raw_scat_3d.set_alpha(1.0)
            title.set_text(f"Step 1: Raw path ({len(raw_path)} points)")
        elif f < p2_end:
            rdp_3d.set_alpha(1.0)
            rdp_3d.set_data(rdp_path[:, 0], rdp_path[:, 1])
            rdp_3d.set_3d_properties(rdp_path[:, 2])
            rdp_top.set_alpha(1.0)
            rdp_top.set_data(rdp_path[:, 0], rdp_path[:, 1])
            rdp_side.set_alpha(1.0)
            rdp_side.set_data(rdp_path[:, 0], rdp_path[:, 2])
            raw_3d.set_alpha(0.2)
            raw_top.set_alpha(0.2)
            raw_side.set_alpha(0.2)
            raw_scat_3d.set_alpha(0.2)
            title.set_text(f"Step 2: RDP simplified ({len(rdp_path)} points)")
        elif f < p3_end:
            sm_3d.set_alpha(1.0)
            sm_3d.set_data(smooth_path[:, 0], smooth_path[:, 1])
            sm_3d.set_3d_properties(smooth_path[:, 2])
            sm_top.set_alpha(1.0)
            sm_top.set_data(smooth_path[:, 0], smooth_path[:, 1])
            sm_side.set_alpha(1.0)
            sm_side.set_data(smooth_path[:, 0], smooth_path[:, 2])
            rdp_3d.set_alpha(0.2)
            rdp_top.set_alpha(0.2)
            rdp_side.set_alpha(0.2)
            title.set_text(f"Step 3: Cubic resample ({len(smooth_path)} points)")
        else:
            fi = f - p3_end
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Quadrotor Following Smoothed Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
