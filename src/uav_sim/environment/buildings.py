# Erwin Lejeune - 2026-02-17
"""Procedural building generators for urban environments."""

from __future__ import annotations

import numpy as np

from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.environment.world import World


def add_city_grid(
    world: World,
    n_blocks: tuple[int, int] = (3, 3),
    block_size: float = 8.0,
    street_width: float = 4.0,
    height_range: tuple[float, float] = (5.0, 25.0),
    seed: int | None = None,
) -> list[BoxObstacle]:
    """Add a grid of box-shaped buildings to *world*.

    Returns the list of created obstacles for visualisation.
    """
    rng = np.random.default_rng(seed)
    buildings: list[BoxObstacle] = []
    pitch = block_size + street_width
    for ix in range(n_blocks[0]):
        for iy in range(n_blocks[1]):
            x0 = ix * pitch + street_width
            y0 = iy * pitch + street_width
            h = rng.uniform(*height_range)
            b = BoxObstacle(
                min_corner=np.array([x0, y0, 0.0]),
                max_corner=np.array([x0 + block_size, y0 + block_size, h]),
            )
            buildings.append(b)
            world.add_obstacle(b)
    return buildings


def add_urban_buildings(
    world: World,
    world_size: float = 30.0,
    n_buildings: int = 6,
    height_range: tuple[float, float] = (8.0, 22.0),
    size_range: tuple[float, float] = (2.0, 5.0),
    margin: float = 2.0,
    seed: int = 42,
) -> list[BoxObstacle]:
    """Scatter realistic box buildings across the world.

    Avoids placing buildings in the corners (start/goal areas) and ensures
    a minimum gap between them for path-planning feasibility.
    """
    rng = np.random.default_rng(seed)
    buildings: list[BoxObstacle] = []
    for _ in range(n_buildings * 10):
        if len(buildings) >= n_buildings:
            break
        w = rng.uniform(*size_range)
        d = rng.uniform(*size_range)
        h = rng.uniform(*height_range)
        x0 = rng.uniform(margin, world_size - margin - w)
        y0 = rng.uniform(margin, world_size - margin - d)
        # keep start (2,2) and goal (~28,28) areas clear
        cx, cy = x0 + w / 2, y0 + d / 2
        if cx < 5 and cy < 5:
            continue
        if cx > world_size - 5 and cy > world_size - 5:
            continue

        lo = np.array([x0, y0, 0.0])
        hi = np.array([x0 + w, y0 + d, h])
        # check overlap with existing
        ok = True
        for b in buildings:
            if _boxes_overlap_xy(lo, hi, b.min_corner, b.max_corner, gap=1.5):
                ok = False
                break
        if ok:
            b = BoxObstacle(min_corner=lo, max_corner=hi)
            buildings.append(b)
            world.add_obstacle(b)
    return buildings


def _boxes_overlap_xy(
    lo1: np.ndarray,
    hi1: np.ndarray,
    lo2: np.ndarray,
    hi2: np.ndarray,
    gap: float = 0.0,
) -> bool:
    return not (
        hi1[0] + gap < lo2[0]
        or hi2[0] + gap < lo1[0]
        or hi1[1] + gap < lo2[1]
        or hi2[1] + gap < lo1[1]
    )
