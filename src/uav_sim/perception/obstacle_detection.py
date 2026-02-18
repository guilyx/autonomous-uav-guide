# Erwin Lejeune - 2026-02-17
"""Simple obstacle detection from range measurements."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RangeObstacleDetector:
    """Detect obstacle segments from a 2-D lidar scan.

    Groups consecutive beams below a threshold into obstacle clusters
    and returns their (angle, range) centroids.
    """

    def __init__(self, threshold: float = 10.0, min_cluster_size: int = 3) -> None:
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size

    def detect(
        self,
        ranges: NDArray[np.floating],
        angles: NDArray[np.floating],
    ) -> list[tuple[float, float]]:
        """Return list of (angle, range) for detected obstacle centroids."""
        hits = ranges < self.threshold
        clusters: list[list[int]] = []
        current: list[int] = []
        for i, h in enumerate(hits):
            if h:
                current.append(i)
            else:
                if len(current) >= self.min_cluster_size:
                    clusters.append(current)
                current = []
        if len(current) >= self.min_cluster_size:
            clusters.append(current)

        centroids = []
        for c in clusters:
            mean_angle = float(np.mean(angles[c]))
            mean_range = float(np.mean(ranges[c]))
            centroids.append((mean_angle, mean_range))
        return centroids
