# Erwin Lejeune - 2026-02-15
"""High-quality costmap visualisation utilities.

Inspired by the `simple_autonomous_car` costmap renderer: masked free space,
proper colormaps with gradient-visible inflation, and colorbar legends.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray


def draw_costmap_heatmap(
    ax: Any,
    grid: NDArray[np.floating],
    extent: tuple[float, float, float, float],
    *,
    cmap: str = "hot_r",
    alpha: float = 0.65,
    mask_free: bool = True,
    vmin: float = 0.0,
    vmax: float | None = None,
    add_colorbar: bool = True,
    colorbar_label: str = "Cost",
) -> list[Artist]:
    """Render a 2-D costmap slice as a masked heatmap on *ax*.

    Parameters
    ----------
    grid : 2-D array of cost values (rows = Y, cols = X).
    extent : (xmin, xmax, ymin, ymax) for ``imshow``.
    mask_free : hide cells with cost == 0 so the map shows through.
    """
    arts: list[Artist] = []
    g = np.asarray(grid, dtype=np.float64)
    if mask_free:
        g = np.ma.masked_where(g < 1e-6, g)

    vmax = vmax if vmax is not None else max(float(np.nanmax(g)), 1.0)
    im = ax.imshow(
        g,
        origin="lower",
        extent=extent,
        cmap=cmap,
        alpha=alpha,
        norm=Normalize(vmin=vmin, vmax=vmax),
        interpolation="nearest",
        zorder=2,
    )
    arts.append(im)

    if add_colorbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
        cb.set_label(colorbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    return arts


def draw_costmap_surface(
    ax: Axes3D,
    grid: NDArray[np.floating],
    extent: tuple[float, float, float, float],
    z_offset: float = 0.0,
    *,
    cmap: str = "hot_r",
    alpha: float = 0.5,
    z_scale: float = 1.0,
) -> list[Artist]:
    """Render a costmap as a 3-D surface (cost → height)."""
    arts: list[Artist] = []
    g = np.asarray(grid, dtype=np.float64)

    xmin, xmax, ymin, ymax = extent
    X = np.linspace(xmin, xmax, g.shape[1])
    Y = np.linspace(ymin, ymax, g.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = g * z_scale + z_offset

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        zorder=1,
    )
    arts.append(surf)
    return arts


def draw_occupancy_overlay(
    ax: Any,
    occupancy: NDArray[np.floating],
    extent: tuple[float, float, float, float],
    *,
    alpha: float = 0.4,
) -> list[Artist]:
    """Draw binary occupancy as black (occupied) / transparent (free)."""
    arts: list[Artist] = []
    g = np.asarray(occupancy, dtype=np.float64)
    g_masked = np.ma.masked_where(g < 0.5, g)
    im = ax.imshow(
        g_masked,
        origin="lower",
        extent=extent,
        cmap="Greys",
        alpha=alpha,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=2,
    )
    arts.append(im)
    return arts


def create_four_panel_figure(
    title: str = "UAV Simulation",
    world_size: float = 30.0,
    figsize: tuple[float, float] = (18, 9),
) -> tuple[Figure, Axes3D, Any, Any, Any]:
    """Create a 4-panel figure: 3D | top-down | side | sensor.

    Returns (fig, ax3d, ax_top, ax_side, ax_sensor).
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1], hspace=0.30, wspace=0.28)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 1])
    ax_sensor = fig.add_subplot(gs[:, 2])

    fig.suptitle(title, fontsize=13)

    ax3d.set_xlim(0, world_size)
    ax3d.set_ylim(0, world_size)
    ax3d.set_zlim(0, world_size)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")

    for ax, ylabel, title_s in [
        (ax_top, "Y [m]", "Top View (XY)"),
        (ax_side, "Z [m]", "Side View (XZ)"),
    ]:
        ax.set_xlim(0, world_size)
        ax.set_ylim(0, world_size)
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title_s, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    ax_top.set_aspect("equal")

    ax_sensor.set_title("Sensor Data", fontsize=9)
    ax_sensor.tick_params(labelsize=7)

    return fig, ax3d, ax_top, ax_side, ax_sensor
