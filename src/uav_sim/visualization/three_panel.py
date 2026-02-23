# Erwin Lejeune - 2026-02-19
"""Reusable 2-panel + data layout for UAV simulation visualisations.

Provides a standardised figure with:
  - **3D** perspective (left)
  - **2D top-down** XY view (upper-right)
  - **Data** axes for telemetry plots (lower-right)

.. note:: The class name is kept as ``ThreePanelViz`` for backwards
   compatibility even though the side-view panel has been replaced
   by a data panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

from uav_sim.environment.obstacles import BoxObstacle, CylinderObstacle, Obstacle
from uav_sim.visualization.theme import (
    C_BUILDING_EDGE,
    C_BUILDING_FACE,
    C_GOAL,
    C_PATH,
    C_START,
    C_TRAIL,
    apply_theme,
)
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
    draw_quadrotor_3d,
)


@dataclass
class ThreePanelViz:
    """Standardised 3D + top-down + data figure for UAV simulations.

    Parameters
    ----------
    title : Figure super-title.
    world_size : Axis limits ``[0, world_size]`` in X, Y, Z.
    figsize : Figure dimensions in inches.
    """

    title: str = "UAV Simulation"
    world_size: float = 30.0
    z_max: float | None = None
    figsize: tuple[float, float] = (16, 8)

    fig: Figure = field(init=False, repr=False)
    ax3d: Axes3D = field(init=False, repr=False)
    ax_top: Any = field(init=False, repr=False)
    ax_data: Any = field(init=False, repr=False)
    ax_side: Any = field(init=False, repr=False)
    _vehicle_arts_3d: list[Artist] = field(init=False, default_factory=list)
    _vehicle_arts_top: list[Artist] = field(init=False, default_factory=list)
    _vehicle_arts_side: list[Artist] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        apply_theme()
        zm = self.z_max if self.z_max is not None else self.world_size
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(
            2,
            2,
            width_ratios=[1.3, 1],
            hspace=0.30,
            wspace=0.28,
        )
        self.ax3d = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.ax_top = self.fig.add_subplot(gs[0, 1])
        self.ax_data = self.fig.add_subplot(gs[1, 1])
        self.fig.suptitle(self.title, fontsize=13)

        # Hidden side axes for backward compat — accepts plot calls silently
        self.ax_side = self.fig.add_axes([0, 0, 0.001, 0.001])
        self.ax_side.set_visible(False)

        # 3D
        self.ax3d.set_xlim(0, self.world_size)
        self.ax3d.set_ylim(0, self.world_size)
        self.ax3d.set_zlim(0, zm)
        self.ax3d.set_xlabel("X [m]")
        self.ax3d.set_ylabel("Y [m]")
        self.ax3d.set_zlabel("Z [m]")

        # Top-down (XY)
        self.ax_top.set_xlim(0, self.world_size)
        self.ax_top.set_ylim(0, self.world_size)
        self.ax_top.set_xlabel("X [m]", fontsize=8)
        self.ax_top.set_ylabel("Y [m]", fontsize=8)
        self.ax_top.set_title("Top View (XY)", fontsize=9)
        self.ax_top.set_aspect("equal")
        self.ax_top.grid(True, alpha=0.2)
        self.ax_top.tick_params(labelsize=7)

        # Data panel (default: blank with grid)
        self.ax_data.grid(True, alpha=0.2)
        self.ax_data.tick_params(labelsize=7)
        self.ax_data.set_title("Data", fontsize=9)

    # ── static geometry ────────────────────────────────────────────────

    def draw_buildings(self, obstacles: list[Obstacle]) -> None:
        """Render box/cylinder obstacles in 3D and top-down views."""
        for obs in obstacles:
            if isinstance(obs, BoxObstacle):
                lo, hi = obs.min_corner, obs.max_corner
                _draw_box_3d(self.ax3d, lo, hi)
                _draw_box_top(self.ax_top, lo, hi)
            elif isinstance(obs, CylinderObstacle):
                _draw_cyl_top(self.ax_top, obs)

    def mark_start_goal(
        self,
        start: NDArray[np.floating],
        goal: NDArray[np.floating],
    ) -> None:
        """Plot start/goal markers on 3D and top-down views."""
        self.ax3d.scatter(*start, c=C_START, s=120, marker="^", label="Start", zorder=5)
        self.ax3d.scatter(*goal, c=C_GOAL, s=120, marker="v", label="Goal", zorder=5)
        self.ax3d.legend(fontsize=7, loc="upper left")
        self.ax_top.plot(start[0], start[1], "^", color=C_START, ms=10, zorder=5)
        self.ax_top.plot(goal[0], goal[1], "v", color=C_GOAL, ms=10, zorder=5)

    def draw_path(
        self,
        path: NDArray[np.floating],
        color: str = C_PATH,
        lw: float = 1.5,
        alpha: float = 0.6,
        label: str = "",
    ) -> None:
        """Draw a static path/trajectory on 3D and top-down views."""
        kw: dict = {"color": color, "lw": lw, "alpha": alpha}
        if label:
            kw["label"] = label
        self.ax3d.plot(path[:, 0], path[:, 1], path[:, 2], **kw)
        self.ax_top.plot(path[:, 0], path[:, 1], **kw)

    # ── per-frame update helpers ────────────────────────────────────────

    def create_trail_artists(
        self,
        color: str = C_TRAIL,
        lw: float = 1.5,
    ) -> dict[str, Any]:
        """Create empty line artists for the live drone trail."""
        (t3d,) = self.ax3d.plot([], [], [], color=color, lw=lw)
        (t_top,) = self.ax_top.plot([], [], color=color, lw=lw)
        return {"3d": t3d, "top": t_top}

    def update_trail(
        self,
        artists: dict[str, Any],
        positions: NDArray[np.floating],
        k: int,
    ) -> None:
        """Update trail lines up to frame index *k*."""
        pos = positions[:k]
        if len(pos) == 0:
            return
        artists["3d"].set_data(pos[:, 0], pos[:, 1])
        artists["3d"].set_3d_properties(pos[:, 2])
        artists["top"].set_data(pos[:, 0], pos[:, 1])

    def update_vehicle(
        self,
        position: NDArray[np.floating],
        euler: NDArray[np.floating],
        size: float = 1.5,
    ) -> None:
        """Draw the quadrotor at given pose, clearing previous artists."""
        from uav_sim.vehicles.multirotor.quadrotor import Quadrotor

        clear_vehicle_artists(self._vehicle_arts_3d)
        clear_vehicle_artists(self._vehicle_arts_top)

        R = Quadrotor.rotation_matrix(*euler)
        self._vehicle_arts_3d.extend(draw_quadrotor_3d(self.ax3d, position, R, size=size))
        self._vehicle_arts_top.extend(
            draw_quadrotor_2d(
                self.ax_top,
                position[:2],
                euler[2],
                size=size,
                arm_lw=1.0,
                motor_size=10,
            )
        )

    def setup_data_axes(
        self,
        xlabel: str = "Time [s]",
        ylabel: str = "",
        title: str = "Data",
    ) -> Any:
        """Configure the data panel. Returns the axes."""
        self.ax_data.set_xlabel(xlabel, fontsize=8)
        self.ax_data.set_ylabel(ylabel, fontsize=8)
        self.ax_data.set_title(title, fontsize=9)
        return self.ax_data


# ── box drawing helpers ──────────────────────────────────────────────────────


def _draw_box_3d(ax: Axes3D, lo: NDArray, hi: NDArray) -> None:
    """Render an axis-aligned box as semi-transparent faces on 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    x0, y0, z0 = lo
    x1, y1, z1 = hi
    verts = [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],
    ]
    poly = Poly3DCollection(
        verts,
        alpha=0.18,
        facecolor=C_BUILDING_FACE,
        edgecolor=C_BUILDING_EDGE,
        linewidth=0.3,
    )
    ax.add_collection3d(poly)


def _draw_box_top(ax: Any, lo: NDArray, hi: NDArray) -> None:
    """Draw box footprint on the top-down XY view."""
    import matplotlib.patches as patches

    w, h = hi[0] - lo[0], hi[1] - lo[1]
    rect = patches.Rectangle(
        (lo[0], lo[1]),
        w,
        h,
        linewidth=0.5,
        edgecolor=C_BUILDING_EDGE,
        facecolor=C_BUILDING_FACE,
        alpha=0.35,
    )
    ax.add_patch(rect)


def _draw_cyl_top(ax: Any, obs: CylinderObstacle) -> None:
    circle = plt.Circle(
        (obs.centre[0], obs.centre[1]),
        obs.radius,
        color=C_BUILDING_FACE,
        alpha=0.35,
    )
    ax.add_patch(circle)
