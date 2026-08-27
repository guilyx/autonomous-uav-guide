# Erwin Lejeune - 2026-02-17
"""Fixed-wing aircraft models, airframe presets and trim solving."""

from flybots.vehicles.fixed_wing.aerodynamics import (
    AeroCoefficients,
    AeroState,
    AeroWrench,
    PropulsionParams,
)
from flybots.vehicles.fixed_wing.fixed_wing import FixedWing, FixedWingParams
from flybots.vehicles.fixed_wing.presets import (
    FixedWingPreset,
    create_fixed_wing,
    get_fixed_wing_params,
)
from flybots.vehicles.fixed_wing.trim import TrimError, TrimPoint, compute_trim

__all__ = [
    "AeroCoefficients",
    "AeroState",
    "AeroWrench",
    "FixedWing",
    "FixedWingParams",
    "FixedWingPreset",
    "PropulsionParams",
    "TrimError",
    "TrimPoint",
    "compute_trim",
    "create_fixed_wing",
    "get_fixed_wing_params",
]
