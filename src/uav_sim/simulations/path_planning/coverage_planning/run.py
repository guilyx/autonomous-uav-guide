# Erwin Lejeune - 2026-02-18
"""Coverage Path Planning: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: boustrophedon decomposition of the survey region.
           Shows the sweep rows being laid out progressively, with the
           sensor swath footprint highlighted.
Phase 2 — Platform: quadrotor takeoff → pure-pursuit along coverage
           path → land.

Reference: H. Choset, "Coverage of Known Spaces: The Boustrophedon
Cellular Decomposition," Autonomous Robots, 2000.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_planning.coverage_planner import CoveragePathPlanner, CoverageRegion
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 10.0
SWATH = 5.0


def main() -> None:
    default_world()

    region = CoverageRegion(
        origin=np.array([3.0, 3.0]),
        width=24.0,
        height=24.0,
        altitude=CRUISE_ALT,
    )

    planner = CoveragePathPlanner(swath_width=SWATH, overlap=0.15, margin=1.5, points_per_row=25)
    coverage_path = planner.plan(region)
    coverage_pct = planner.estimated_coverage(region)

    # Phase 2: fly
    quad = Quadrotor()
    quad.reset(position=np.array([coverage_path[0, 0], coverage_path[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    flight_states = fly_mission(
        quad,
        ctrl,
        coverage_path,
        cruise_alt=CRUISE_ALT,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.0,
        landing_duration=2.0,
        loiter_duration=0.3,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation ─────────────────────────────────────────────────────
    n_path_pts = len(coverage_path)
    path_step = max(1, n_path_pts // 60)
    path_frames = list(range(0, n_path_pts, path_step))
    n_pf = len(path_frames)

    fly_step = max(1, len(flight_pos) // 80)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_ff = len(fly_frames)
    total = n_pf + n_ff

    viz = ThreePanelViz(title="Coverage Path Planning — Boustrophedon", world_size=WORLD_SIZE)

    # Draw survey region outline
    rx, ry = region.origin
    for ax2 in [viz.ax_top]:
        rect = mpatches.Rectangle(
            (rx, ry),
            region.width,
            region.height,
            fill=False,
            edgecolor="green",
            lw=2,
            linestyle="--",
            label="Survey Region",
        )
        ax2.add_patch(rect)
    viz.ax3d.plot(
        [rx, rx + region.width, rx + region.width, rx, rx],
        [ry, ry, ry + region.height, ry + region.height, ry],
        [CRUISE_ALT] * 5,
        "g--",
        lw=1.5,
        alpha=0.6,
    )

    # Coverage path — progressively revealed
    (path_3d,) = viz.ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.7, label="Coverage Path")
    (path_top,) = viz.ax_top.plot([], [], "b-", lw=1.0, alpha=0.7)
    (path_side,) = viz.ax_side.plot([], [], "b-", lw=1.0, alpha=0.7)

    # Swath footprint patches (top-down view)
    swath_patches: list[mpatches.Rectangle] = []

    fly_trail = viz.create_trail_artists()
    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Phase 1: Computing Coverage Path")

    anim = SimAnimator("coverage_planning", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < n_pf:
            k = path_frames[f]
            pts = coverage_path[: k + 1]
            path_3d.set_data(pts[:, 0], pts[:, 1])
            path_3d.set_3d_properties(pts[:, 2])
            path_top.set_data(pts[:, 0], pts[:, 1])
            path_side.set_data(pts[:, 0], pts[:, 2])

            # Add swath band for each new row segment
            if len(swath_patches) < k // planner.points_per_row + 1:
                # Determine which row we're on
                row_idx = len(swath_patches)
                step = SWATH * (1.0 - planner.overlap)
                y_center = region.origin[1] + planner.margin + row_idx * step
                half_s = SWATH / 2
                rect = mpatches.Rectangle(
                    (region.origin[0] + planner.margin, y_center - half_s),
                    region.width - 2 * planner.margin,
                    SWATH,
                    facecolor="cyan",
                    alpha=0.1,
                    edgecolor="deepskyblue",
                    lw=0.5,
                )
                viz.ax_top.add_patch(rect)
                swath_patches.append(rect)

            pct = int(100 * (f + 1) / n_pf)
            title.set_text(
                f"Phase 1: Boustrophedon — {pct}% (est. coverage {coverage_pct * 100:.0f}%)"
            )
        else:
            fi = f - n_pf
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Flying Coverage Pattern")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
