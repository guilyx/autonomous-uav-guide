# Erwin Lejeune - 2026-02-20
"""Shared constants and helpers for all UAV simulations.

Provides a single source of truth for standard durations, frame budgets,
and reusable reference-trajectory generators so every simulation looks
and feels consistent.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.environment import default_world
from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.path_planning.plan_through_obstacles import plan_through_obstacles

# ── timing / animation defaults ──────────────────────────────────────────────
STANDARD_DURATION: float = 30.0
STANDARD_FPS: int = 25
MAX_FRAMES: int = 100
WORLD_SIZE: float = 30.0
CRUISE_ALT: float = 12.0

# ── figure-8 reference (for control / tracking sims) ────────────────────────
DEFAULT_CENTER = np.array([15.0, 15.0])
DEFAULT_RX: float = 6.0
DEFAULT_RY: float = 4.0
DEFAULT_ALT: float = 12.0
DEFAULT_OMEGA: float = 0.25


def figure_8_ref(
    t: float,
    center: NDArray[np.floating] = DEFAULT_CENTER,
    rx: float = DEFAULT_RX,
    ry: float = DEFAULT_RY,
    alt: float = DEFAULT_ALT,
    omega: float = DEFAULT_OMEGA,
    alt_amp: float = 1.5,
    alt_freq: float = 0.3,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return ``(position, velocity)`` on a figure-8 at time *t*.

    The Lissajous parametrization ``(sin(ωt), sin(2ωt))`` produces the
    characteristic crossing pattern.
    """
    pos = np.array(
        [
            center[0] + rx * np.sin(omega * t),
            center[1] + ry * np.sin(2.0 * omega * t),
            alt + alt_amp * np.sin(alt_freq * t),
        ]
    )
    vel = np.array(
        [
            rx * omega * np.cos(omega * t),
            ry * 2.0 * omega * np.cos(2.0 * omega * t),
            alt_amp * alt_freq * np.cos(alt_freq * t),
        ]
    )
    return pos, vel


def figure_8_path(
    duration: float = STANDARD_DURATION,
    dt: float = 0.1,
    **kwargs,
) -> NDArray[np.floating]:
    """Sample the figure-8 at fixed *dt* and return an ``(N, 3)`` array."""
    times = np.arange(0.0, duration, dt)
    return np.array([figure_8_ref(t, **kwargs)[0] for t in times])


# ── line-to-goal through obstacles (for planning sims) ──────────────────────
START_POS = np.array([3.0, 3.0, CRUISE_ALT])
GOAL_POS = np.array([27.0, 27.0, CRUISE_ALT])


def line_to_goal(
    buildings: list[BoxObstacle] | None = None,
    start: NDArray[np.floating] = START_POS,
    goal: NDArray[np.floating] = GOAL_POS,
    world_size: int = 30,
) -> tuple[NDArray[np.floating], list[BoxObstacle]]:
    """Plan obstacle-aware path from *start* to *goal*.

    Returns ``(path, buildings)``; if no buildings given uses
    ``default_world()``.
    """
    if buildings is None:
        _, buildings = default_world()
    path = plan_through_obstacles(buildings, start, goal, world_size=world_size)
    if path is None:
        path = np.array([start, goal])
    return path, buildings


# ── costmap overlay helper ───────────────────────────────────────────────────
COSTMAP_CMAP = "RdYlGn_r"


def draw_costmap_overlay(
    ax,
    grid_2d: NDArray[np.floating],
    world_size: float = WORLD_SIZE,
    alpha: float = 0.45,
):
    """Draw a 2D costmap on *ax* using the standard ``RdYlGn_r`` colormap.

    Green = free, yellow = inflated, red = occupied.
    Returns the ``AxesImage`` for later ``set_data`` updates.
    """
    extent = [0, world_size, 0, world_size]
    return ax.imshow(
        grid_2d.T,
        origin="lower",
        extent=extent,
        cmap=COSTMAP_CMAP,
        vmin=0,
        vmax=1,
        alpha=alpha,
        zorder=0,
    )


# ── frame indexing helpers ───────────────────────────────────────────────────
def frame_indices(n_steps: int, max_frames: int = MAX_FRAMES) -> list[int]:
    """Return evenly-spaced frame indices capped at *max_frames*."""
    skip = max(1, n_steps // max_frames)
    return list(range(0, n_steps, skip))
