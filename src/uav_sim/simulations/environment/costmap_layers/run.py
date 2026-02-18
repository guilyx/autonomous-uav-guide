# Erwin Lejeune - 2026-02-18
"""Multi-layer costmap: occupancy + inflation + social in a 30 m urban world.

Four panels:
  - **3D** scene with buildings and a flying dynamic-agent drone.
  - **Raw occupancy** grid (top-down).
  - **Inflated** costmap (top-down).
  - **Social layer** (dynamic, top-down) updating with the moving agent.

Reference: D. V. Lu et al., "Layered Costmaps for Context-Sensitive
Navigation," IROS, 2014.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.costmap import InflationLayer, LayeredCostmap, OccupancyGrid
from uav_sim.costmap.social_layer import SocialLayer
from uav_sim.environment import default_world
from uav_sim.environment.buildings import add_city_grid
from uav_sim.environment.world import DynamicAgent
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import _draw_box_3d
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0


def main() -> None:
    world, buildings = default_world()
    buildings = add_city_grid(world, n_blocks=(2, 2), height_range=(5.0, 15.0), seed=42)

    grid = OccupancyGrid(
        resolution=0.5, bounds_min=np.zeros(3), bounds_max=np.array([WORLD_SIZE, WORLD_SIZE, 0.0])
    )
    grid.from_world(world)
    inflation = InflationLayer(inflation_radius=2.5, cost_scaling=2.0)
    social = SocialLayer(max_radius=4.0, speed_scale=1.5, amplitude=0.8)

    costmap = LayeredCostmap(grid, inflation=inflation)
    costmap.update()

    n_frames = 120
    dt = 0.2
    agent_path = np.column_stack(
        [
            np.linspace(3.0, 27.0, n_frames),
            np.linspace(25.0, 5.0, n_frames),
            np.full(n_frames, 12.0),
        ]
    )
    agent_vels = np.diff(agent_path, axis=0, prepend=agent_path[:1]) / dt

    raw_grid = grid.grid.copy()
    inflated = inflation.apply(grid)
    social_layers = []
    for i in range(n_frames):
        agent = DynamicAgent(position=agent_path[i], velocity=agent_vels[i], radius=0.5)
        world._dynamic_agents = [agent]
        social_cost = social.apply(grid, world)
        social_layers.append(
            social_cost[:, :, 0].copy() if social_cost.ndim == 3 else social_cost.copy()
        )

    raw_2d = raw_grid[:, :, 0] if raw_grid.ndim == 3 else raw_grid
    infl_2d = inflated[:, :, 0] if inflated.ndim == 3 else inflated

    # ── Layout: 3D scene + 3 costmap panels ────────────────────────────
    anim = SimAnimator("costmap_layers", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(18, 8))
    anim._fig = fig
    gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1], hspace=0.30, wspace=0.25)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_infl = fig.add_subplot(gs[0, 2])
    ax_soc = fig.add_subplot(gs[1, 1])
    ax_comp = fig.add_subplot(gs[1, 2])
    fig.suptitle("Multi-Layer Costmap — Occupancy | Inflation | Social", fontsize=13)

    # 3D scene
    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    for b in buildings:
        lo, hi = b.bounding_box()
        _draw_box_3d(ax3d, lo, hi)

    (trail3d,) = ax3d.plot([], [], [], "c-", lw=1.2, alpha=0.6)
    (agent_trail3d,) = ax3d.plot([], [], [], "m-", lw=1, alpha=0.5)

    extent = [0, WORLD_SIZE, 0, WORLD_SIZE]
    for ax, title in [
        (ax_raw, "Raw Occupancy"),
        (ax_infl, "Inflated"),
        (ax_soc, "Social Layer"),
        (ax_comp, "Composite"),
    ]:
        ax.set_aspect("equal")
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_xlabel("X [m]", fontsize=7)
        ax.set_ylabel("Y [m]", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)

    ax_raw.imshow(raw_2d.T, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    ax_infl.imshow(infl_2d.T, origin="lower", extent=extent, cmap="hot_r", vmin=0, vmax=1)
    im_soc = ax_soc.imshow(
        social_layers[0].T, origin="lower", extent=extent, cmap="YlOrRd", vmin=0, vmax=1
    )
    composite_0 = np.clip(infl_2d + social_layers[0], 0, 1)
    im_comp = ax_comp.imshow(
        composite_0.T, origin="lower", extent=extent, cmap="hot_r", vmin=0, vmax=1
    )

    (ad1,) = ax_raw.plot([], [], "co", ms=6)
    (ad2,) = ax_infl.plot([], [], "co", ms=6)
    (ad3,) = ax_soc.plot([], [], "co", ms=6)
    (ad4,) = ax_comp.plot([], [], "co", ms=6)

    ego_pos = np.array([15.0, 15.0, 12.0])
    veh3d: list = []
    veh_agent: list = []

    def update(f: int) -> None:
        ap = agent_path[f]

        trail3d.set_data(agent_path[:f, 0], agent_path[:f, 1])
        trail3d.set_3d_properties(agent_path[:f, 2])

        clear_vehicle_artists(veh3d)
        R = Quadrotor.rotation_matrix(0, 0, 0)
        veh3d.extend(draw_quadrotor_3d(ax3d, ego_pos, R, size=1.5))

        clear_vehicle_artists(veh_agent)
        yaw_ag = np.arctan2(agent_vels[f, 1], agent_vels[f, 0])
        R_ag = Quadrotor.rotation_matrix(0, 0, yaw_ag)
        veh_agent.extend(
            draw_quadrotor_3d(ax3d, ap, R_ag, size=1.0, arm_colors=("magenta", "purple"))
        )

        for dot in [ad1, ad2, ad3, ad4]:
            dot.set_data([ap[0]], [ap[1]])

        im_soc.set_data(social_layers[f].T)
        composite_f = np.clip(infl_2d + social_layers[f], 0, 1)
        im_comp.set_data(composite_f.T)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
