"""Speed-aware additive layer for 2D costmaps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class VelocityCostLayer:
    """Increase local cost proportionally to ego speed.

    The layer is blended with an existing costmap by adding a bounded penalty.
    This keeps occupied cells occupied while increasing caution in free space
    when speed is high.
    """

    def __init__(self, max_speed: float = 6.0, max_penalty: float = 0.35) -> None:
        self.max_speed = max(1e-6, float(max_speed))
        self.max_penalty = float(np.clip(max_penalty, 0.0, 1.0))

    def apply(
        self,
        base_costmap: NDArray[np.floating],
        speed: float,
    ) -> NDArray[np.floating]:
        speed_ratio = float(np.clip(speed / self.max_speed, 0.0, 1.0))
        penalty = self.max_penalty * speed_ratio
        return np.clip(base_costmap + penalty * (1.0 - base_costmap), 0.0, 1.0).astype(np.float32)
