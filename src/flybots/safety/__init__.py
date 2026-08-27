# Erwin Lejeune - 2026-08-27
"""Control barrier functions and the QP that enforces them.

A barrier defines a safe set; the filter wraps whatever controller is
already flying and changes its command only when a barrier is about to be
violated, by the smallest amount that keeps it satisfied.
"""

from .barriers import (
    AltitudeFloorBarrier,
    Barrier,
    BarrierRows,
    ConnectivityBarrier,
    GeofenceBoxBarrier,
    SafeDistanceBarrier,
    SpeedLimitBarrier,
    SphereObstacleBarrier,
)
from .filter import FilterReport, SafetyFilter
from .qp import InfeasibleQPError, solve_least_distance, solve_safety_qp

__all__ = [
    "AltitudeFloorBarrier",
    "Barrier",
    "BarrierRows",
    "ConnectivityBarrier",
    "FilterReport",
    "GeofenceBoxBarrier",
    "InfeasibleQPError",
    "SafeDistanceBarrier",
    "SafetyFilter",
    "SpeedLimitBarrier",
    "SphereObstacleBarrier",
    "solve_least_distance",
    "solve_safety_qp",
]
