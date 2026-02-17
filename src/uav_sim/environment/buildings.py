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
