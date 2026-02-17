# Erwin Lejeune - 2026-02-17
"""Multi-layer costmap visualisation: occupancy + inflation + social.

Shows three costmap layers side-by-side:
1. Raw occupancy grid from building obstacles
2. Inflated costmap (safe-distance buffer)
3. Social layer with a moving dynamic agent

A quadrotor navigates through the environment using A* on the inflated map,
while the social layer updates dynamically with a moving obstacle drone.

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
from uav_sim.environment import World
from uav_sim.environment.buildings import add_city_grid
from uav_sim.environment.world import DynamicAgent
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
)

matplotlib.use("Agg")


def main() -> None:
    world = World(bounds_min=np.zeros(3), bounds_max=np.array([30.0, 30.0, 20.0]))
    add_city_grid(world, n_blocks=(2, 2), height_range=(5.0, 15.0), seed=42)

    grid = OccupancyGrid(
        resolution=0.5,
        bounds_min=np.zeros(3),
        bounds_max=np.array([30.0, 30.0, 0.0]),
    )
    grid.from_world(world)
    inflation = InflationLayer(inflation_radius=2.5, cost_scaling=2.0)
    social = SocialLayer(max_radius=4.0, speed_scale=1.5, amplitude=0.8)

    costmap = LayeredCostmap(grid, inflation=inflation)
    costmap.update()

    # ── Dynamic agent moving through the scene ────────────────────────────
    n_frames = 120
    dt = 0.2
    agent_path = np.column_stack(
        [
            np.linspace(3.0, 27.0, n_frames),
            np.linspace(25.0, 5.0, n_frames),
            np.zeros(n_frames),
        ]
    )
    agent_vels = np.diff(agent_path, axis=0, prepend=agent_path[:1]) / dt

    # ── Pre-compute layers for each frame ─────────────────────────────────
    raw_grid = grid.grid.copy()
    inflated = inflation.apply(grid)
    social_layers = []
    for i in range(n_frames):
        agent = DynamicAgent(
            position=agent_path[i],
            velocity=agent_vels[i],
            radius=0.5,
        )
        world._dynamic_agents = [agent]
        social_cost = social.apply(grid, world)
        if social_cost.ndim == 3:
            social_layers.append(social_cost[:, :, 0].copy())
        else:
            social_layers.append(social_cost.copy())

    # ── Extract 2D grids ──────────────────────────────────────────────────
    if raw_grid.ndim == 3:
        raw_2d = raw_grid[:, :, 0]
        infl_2d = inflated[:, :, 0] if inflated.ndim == 3 else inflated
    else:
        raw_2d = raw_grid
        infl_2d = inflated

    # ── Animation ─────────────────────────────────────────────────────────
    anim = SimAnimator("costmap_layers", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(16, 5.5))
    anim._fig = fig
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    fig.suptitle("Multi-Layer Costmap — Occupancy | Inflation | Social", fontsize=13)

    extent = [0, 30, 0, 30]
    for ax, title in [(ax1, "Raw Occupancy"), (ax2, "Inflated"), (ax3, "Social Layer")]:
        ax.set_aspect("equal")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Y [m]", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)

    ax1.imshow(raw_2d.T, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    ax2.imshow(infl_2d.T, origin="lower", extent=extent, cmap="hot_r", vmin=0, vmax=1)
    im3 = ax3.imshow(
        social_layers[0].T, origin="lower", extent=extent, cmap="YlOrRd", vmin=0, vmax=1
    )

    (agent_dot1,) = ax1.plot([], [], "co", ms=8)
    (agent_dot2,) = ax2.plot([], [], "co", ms=8)
    (agent_dot3,) = ax3.plot([], [], "co", ms=8)

    vehicle_arts: list = []

    def update(f):
        clear_vehicle_artists(vehicle_arts)
        ap = agent_path[f, :2]
        agent_dot1.set_data([ap[0]], [ap[1]])
        agent_dot2.set_data([ap[0]], [ap[1]])
        agent_dot3.set_data([ap[0]], [ap[1]])
        im3.set_data(social_layers[f].T)
        # Show a quad tracking the agent at a safe distance
        ego_pos = np.array([15.0, 15.0])
        yaw = np.arctan2(ap[1] - ego_pos[1], ap[0] - ego_pos[0])
        vehicle_arts.extend(draw_quadrotor_2d(ax2, ego_pos, yaw, size=0.8))

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
