# Erwin Lejeune - 2026-02-17
"""Ready-made fixed-wing airframes.

    >>> from flybots.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing
    >>> aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)

Provenance
----------
:attr:`FixedWingPreset.AEROSONDE` uses the published coefficient set from
Beard & McLain, Appendix E, and is the reference airframe for this module.

The other presets are **representative, not measured**. Their geometry and
mass come from the real aircraft, and their aerodynamic coefficients are
scaled from the Aerosonde set with configuration-appropriate adjustments
(for example, a flying wing gets reduced yaw stiffness and pitch damping
because it has no tail boom). They are good enough to fly, tune
controllers against, and teach with — they are not a substitute for wind
tunnel data on a specific airframe.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np

from flybots.vehicles.fixed_wing.aerodynamics import AeroCoefficients, PropulsionParams
from flybots.vehicles.fixed_wing.fixed_wing import FixedWing, FixedWingParams

__all__ = ["FixedWingPreset", "create_fixed_wing", "get_fixed_wing_params"]


class FixedWingPreset(Enum):
    """Catalogue of supported fixed-wing airframes."""

    AEROSONDE = "aerosonde"
    """13.5 kg research UAV — the Beard & McLain reference airframe."""

    SKYWALKER_X8 = "skywalker_x8"
    """3.4 kg flying wing, a common research and FPV platform."""

    MINI_TRAINER = "mini_trainer"
    """0.6 kg foam trainer — slow, docile, fits in a small world."""

    CARGO_UAV = "cargo_uav"
    """25 kg twin-boom cargo aircraft — heavy, high wing loading."""

    CUSTOM = "custom"


_AEROSONDE = FixedWingParams()

# Flying wing: no tail, so pitch damping and yaw stiffness drop sharply and
# the elevons do double duty as elevator and aileron.
_X8_COEFFS = replace(
    AeroCoefficients(),
    CL0=0.0867,
    CLa=4.02,
    CLq=3.87,
    CLde=0.278,
    CD0=0.0197,
    CDa=0.0791,
    CDde=0.0633,
    Cm0=0.0227,
    Cma=-0.583,
    Cmq=-1.34,
    Cmde=-0.271,
    CYb=-0.224,
    CYdr=0.0,
    Clb=-0.0849,
    Clp=-0.4042,
    Clr=0.0555,
    Clda=0.12,
    Cldr=0.0,
    Cnb=0.0283,
    Cnp=-0.0575,
    Cnr=-0.0089,
    Cnda=-0.00339,
    Cndr=0.0,
    alpha_stall=0.3,
    e_oswald=0.85,
)

# Small foam trainer: low Reynolds number, so higher parasitic drag and an
# earlier, softer stall. Generous tail volume makes it very stable.
_TRAINER_COEFFS = replace(
    AeroCoefficients(),
    CL0=0.28,
    CLa=4.6,
    CLq=6.5,
    CLde=0.22,
    CD0=0.055,
    CDa=0.05,
    Cm0=0.02,
    Cma=-1.8,
    Cmq=-20.0,
    Cmde=-0.75,
    Clp=-0.45,
    Clda=0.20,
    Cnb=0.09,
    Cnr=-0.12,
    Cndr=-0.08,
    alpha_stall=0.35,
    stall_sharpness=25.0,
    e_oswald=0.8,
)

_PRESETS: dict[FixedWingPreset, FixedWingParams] = {
    FixedWingPreset.AEROSONDE: _AEROSONDE,
    FixedWingPreset.SKYWALKER_X8: FixedWingParams(
        mass=3.364,
        wing_area=0.75,
        wing_span=2.1,
        chord=0.3571,
        inertia=np.diag([0.335, 0.144, 0.400]),
        rho_air=1.225,
        coeffs=_X8_COEFFS,
        propulsion=PropulsionParams(
            prop_area=0.0314,
            prop_efficiency=1.0,
            k_motor=40.0,
        ),
        cruise_airspeed=18.0,
    ),
    FixedWingPreset.MINI_TRAINER: FixedWingParams(
        mass=0.6,
        wing_area=0.18,
        wing_span=1.0,
        chord=0.18,
        inertia=np.diag([0.0075, 0.0090, 0.0140]),
        rho_air=1.225,
        coeffs=_TRAINER_COEFFS,
        propulsion=PropulsionParams(
            prop_area=0.0122,
            prop_efficiency=1.0,
            k_motor=22.0,
        ),
        cruise_airspeed=12.0,
    ),
    FixedWingPreset.CARGO_UAV: FixedWingParams(
        mass=25.0,
        wing_area=1.2,
        wing_span=4.0,
        chord=0.30,
        inertia=np.diag([2.5, 3.2, 5.0]),
        rho_air=1.225,
        coeffs=replace(AeroCoefficients(), CD0=0.05, Cma=-2.2, Cmq=-32.0),
        propulsion=PropulsionParams(
            prop_area=0.32,
            prop_efficiency=1.0,
            k_motor=75.0,
        ),
        cruise_airspeed=32.0,
    ),
}


def get_fixed_wing_params(preset: FixedWingPreset) -> FixedWingParams:
    """Return the parameter set for a named airframe."""
    if preset is FixedWingPreset.CUSTOM:
        raise ValueError("CUSTOM has no stored parameters — pass overrides to create_fixed_wing.")
    return _PRESETS[preset]


def create_fixed_wing(
    preset: FixedWingPreset = FixedWingPreset.AEROSONDE,
    **overrides,
) -> FixedWing:
    """Create a :class:`FixedWing` from a named preset.

    Keyword arguments override individual :class:`FixedWingParams` fields::

        create_fixed_wing(FixedWingPreset.SKYWALKER_X8, mass=4.0)
    """
    if preset is FixedWingPreset.CUSTOM:
        return FixedWing(FixedWingParams(**overrides))
    base = _PRESETS[preset]
    if not overrides:
        return FixedWing(base)
    return FixedWing(replace(base, **overrides))
