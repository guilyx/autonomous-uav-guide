# Erwin Lejeune - 2026-08-18
"""Guidance laws that sit above an autopilot and steer it along a path.

An autopilot answers "hold this course, altitude and airspeed". A guidance
law answers "which course, altitude and airspeed should I be holding right
now to be on that path". This package supplies the second layer for
fixed-wing aircraft:

* :mod:`~uav_sim.guidance.fixed_wing_paths` — the geometry. Straight-line
  and orbit vector fields, stateless and pure.
* :mod:`~uav_sim.guidance.fixed_wing_mission` — the sequencing. Waypoint
  acceptance, racetracks, and return-to-launch, emitting
  :class:`~uav_sim.control.fixed_wing_autopilot.AutopilotCommand`.

Reference: R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
Practice*, Princeton University Press, 2012, Chapters 10-11.
"""

from uav_sim.guidance.fixed_wing_mission import (
    FixedWingMission,
    LineLeg,
    MissionDiagnostics,
    MissionLeg,
    OrbitLeg,
    orbit_plan,
    racetrack_plan,
    return_to_launch_plan,
    waypoint_plan,
    waypoint_reached,
)
from uav_sim.guidance.fixed_wing_paths import (
    GuidanceError,
    GuidanceGains,
    GuidanceOutput,
    LinePath,
    OrbitDirection,
    OrbitPath,
    minimum_turn_radius,
)

__all__ = [
    "FixedWingMission",
    "GuidanceError",
    "GuidanceGains",
    "GuidanceOutput",
    "LineLeg",
    "LinePath",
    "MissionDiagnostics",
    "MissionLeg",
    "OrbitDirection",
    "OrbitLeg",
    "OrbitPath",
    "minimum_turn_radius",
    "orbit_plan",
    "racetrack_plan",
    "return_to_launch_plan",
    "waypoint_plan",
    "waypoint_reached",
]
