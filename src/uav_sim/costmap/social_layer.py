# Erwin Lejeune - 2026-02-17
"""Dynamic social costmap layer based on agent velocity.

Increases cost in front of moving dynamic agents proportional to their
speed, modelling the social discomfort / collision risk of passing close
to a fast-moving entity.

Reference: P. Trautman, A. Krause, "Unfreezing the Robot: Navigation in
Dense, Interacting Crowds," IROS, 2010.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.environment.world import World


class SocialLayer:
    """Velocity-dependent cost around dynamic agents.

    Parameters
    ----------
    max_radius:
        Maximum influence radius [m].
    speed_scale:
        How much speed amplifies the cost footprint.
    amplitude:
        Peak cost at the agent centre.
    """

    def __init__(
        self,
        max_radius: float = 5.0,
        speed_scale: float = 1.5,
        amplitude: float = 0.9,
    ) -> None:
        self.max_radius = max_radius
        self.speed_scale = speed_scale
        self.amplitude = amplitude

    def apply(self, grid: OccupancyGrid, world: World) -> NDArray[np.floating]:
        """Return a social cost overlay for current dynamic agents."""
        cost = np.zeros_like(grid.grid, dtype=np.float32)
        for agent in world.dynamic_agents:
            speed = float(np.linalg.norm(agent.velocity))
            if speed < 0.01:
                continue
            radius = min(agent.radius + self.speed_scale * speed, self.max_radius)
            centre_cell = grid.world_to_cell(agent.position)
            r_cells = int(radius / grid.resolution) + 1
            slices = []
            for d in range(len(centre_cell)):
                lo = max(0, centre_cell[d] - r_cells)
                hi = min(cost.shape[d], centre_cell[d] + r_cells + 1)
                slices.append(slice(lo, hi))
            region = cost[tuple(slices)]
            idx = np.indices(region.shape)
            offsets = [
                (idx[d] + slices[d].start - centre_cell[d]) * grid.resolution
                for d in range(len(slices))
            ]
            dist = np.sqrt(sum(o**2 for o in offsets))
            mask = dist <= radius
            region[mask] = np.maximum(
                region[mask],
                self.amplitude * (1.0 - dist[mask] / radius),
            )
        return cost
