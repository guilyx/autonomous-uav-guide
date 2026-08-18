# Erwin Lejeune - 2026-02-15
"""Ready-made vehicle configurations for common multirotor platforms.

Each preset provides physically-validated parameters so users can spin up a
realistic simulation with a single call::

    quad = create_quadrotor(VehiclePreset.CRAZYFLIE)
    hex_ = create_multirotor(VehiclePreset.HEX_S550)

Four-rotor presets carry a :class:`QuadrotorParams`, which describes its
geometry as an arm length plus a frame letter. Anything else carries a
:class:`MultirotorParams` with an explicit rotor layout, because "arm
length" stops being a complete description as soon as the rotors are not
all the same distance out at the same height.

Inertias are not guessed. Each is built from the same lumped model — point
masses at the rotor hubs plus a central disc for the body — so that
``I_zz ~ 2 * I_xx`` comes out of the layout rather than being asserted, and
so that changing the arm length in a fork changes the inertia in the
direction it should.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from uav_sim.vehicles.components.allocation import coaxial_layout, x_layout
from uav_sim.vehicles.multirotor.multirotor import Multirotor, MultirotorParams
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor, QuadrotorParams


class VehiclePreset(Enum):
    """Catalogue of supported drone platforms."""

    CRAZYFLIE = "crazyflie"
    DJI_MINI = "dji_mini"
    RACING_250 = "racing_250"
    DJI_MATRICE = "dji_matrice"
    HEX_S550 = "hex_s550"
    OCTO_X8 = "octo_x8"
    CUSTOM = "custom"


# ---------------------------------------------------------------------
# Hexacopter: 550 mm-class flat hex, six 2212 motors on 10-inch props.
# ---------------------------------------------------------------------
# 550 mm wheelbase puts each rotor 275 mm from the centre. Six motor/ESC/
# prop groups of 100 g at that radius, plus a 1.2 kg body treated as a
# 90 mm disc:
#
#   I_zz = sum(m r^2) + 0.5 m_b r_b^2 = 6(0.1)(0.275^2) + 0.5(1.2)(0.09^2)
#   I_xx = sum(m y^2) + 0.25 m_b r_b^2, and sum(sin^2) = n/2 for an even ring
#
# which gives 0.025 / 0.025 / 0.050 to two significant figures. A 2212 on a
# 1045 prop makes about 8.8 N at 9000 rpm (940 rad/s), so k_thrust follows
# from that pair and the airframe lifts three times its own weight.
_HEX_S550 = MultirotorParams(
    mass=1.8,
    inertia=np.diag([0.025, 0.025, 0.050]),
    rotors=x_layout(6, 0.275),
    k_thrust=1.0e-5,
    k_torque=1.6e-7,
    motor_tau=0.03,
    omega_max=940.0,
    drag_coeff=0.12,
    name="hex_s550",
)

# ---------------------------------------------------------------------
# Octocopter: coaxial X8 — four arms, eight rotors, 900 mm-class.
# ---------------------------------------------------------------------
# The interesting airframe of the two. Its eight rotors sit at four
# positions, so the allocation matrix has four redundant columns: it can
# lose a rotor and still fly, but it has no more roll or pitch authority
# than a quadrotor of the same span. The lower rotor of each pair works in
# the wake of the upper one and returns about 85% of its thrust (Leishman,
# Sec. 2.14), which is why an X8 is heavier on power than a flat octo.
_OCTO_X8 = MultirotorParams(
    mass=4.5,
    inertia=np.diag([0.075, 0.075, 0.150]),
    rotors=coaxial_layout(x_layout(4, 0.35), separation=0.12, lower_efficiency=0.85),
    k_thrust=4.0e-5,
    k_torque=9.6e-7,
    motor_tau=0.04,
    omega_max=620.0,
    drag_coeff=0.25,
    name="octo_x8",
)


_QUAD_PRESETS: dict[VehiclePreset, QuadrotorParams] = {
    VehiclePreset.CRAZYFLIE: QuadrotorParams(
        mass=0.027,
        arm_length=0.046,
        inertia=np.diag([1.66e-5, 1.66e-5, 2.96e-5]),
        k_thrust=2.55e-8,
        k_torque=7.94e-10,
        motor_tau=0.02,
        omega_max=2500.0,
        drag_coeff=0.01,
        name="crazyflie",
    ),
    VehiclePreset.DJI_MINI: QuadrotorParams(
        mass=0.249,
        arm_length=0.11,
        inertia=np.diag([6.5e-4, 6.5e-4, 1.2e-3]),
        # k_thrust sized so hover sits at 64 % of omega_max, the same margin
        # the other three presets carry. Momentum theory agrees: for the
        # 4.7-inch propeller this airframe flies, k = C_T rho D^4 / 4 pi^2
        # puts C_T at 0.073, which is where a propeller that size lives.
        k_thrust=4.6e-7,
        k_torque=1.47e-8,  # holds the airframe's torque-to-thrust ratio at 0.032 m
        motor_tau=0.025,
        omega_max=1800.0,
        drag_coeff=0.03,
        name="dji_mini",
    ),
    VehiclePreset.RACING_250: QuadrotorParams(
        mass=1.5,
        arm_length=0.175,
        inertia=np.diag([0.0082, 0.0082, 0.0148]),
        k_thrust=8.55e-6,
        k_torque=1.36e-7,
        motor_tau=0.02,
        omega_max=1100.0,
        drag_coeff=0.1,
        name="racing_250",
    ),
    VehiclePreset.DJI_MATRICE: QuadrotorParams(
        mass=3.6,
        arm_length=0.32,
        inertia=np.diag([0.045, 0.045, 0.080]),
        k_thrust=3.0e-5,
        k_torque=5.0e-7,
        motor_tau=0.03,
        omega_max=800.0,
        drag_coeff=0.15,
        name="dji_matrice",
    ),
}

_MULTI_PRESETS: dict[VehiclePreset, MultirotorParams] = {
    VehiclePreset.HEX_S550: _HEX_S550,
    VehiclePreset.OCTO_X8: _OCTO_X8,
}

_PRESETS: dict[VehiclePreset, QuadrotorParams | MultirotorParams] = {
    **_QUAD_PRESETS,
    **_MULTI_PRESETS,
}


def _apply_overrides(
    base: QuadrotorParams | MultirotorParams,
    overrides: dict[str, object],
) -> QuadrotorParams | MultirotorParams:
    """Return a copy of *base* with the named fields replaced."""
    if not overrides:
        return base
    kw = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    unknown = set(overrides) - set(kw)
    if unknown:
        raise TypeError(
            f"{type(base).__name__} has no field(s) {sorted(unknown)}. Available: {sorted(kw)}"
        )
    kw.update(overrides)
    return type(base)(**kw)


def create_quadrotor(
    preset: VehiclePreset = VehiclePreset.RACING_250,
    **overrides: float,
) -> Quadrotor:
    """Create a :class:`Quadrotor` from a named preset.

    Any keyword argument overrides the corresponding field in
    :class:`QuadrotorParams` (e.g. ``mass=2.0``).

    Parameters
    ----------
    preset : which platform to use. Must be a four-rotor one; use
        :func:`create_multirotor` for the rest.
    **overrides : per-field overrides forwarded to ``QuadrotorParams``.

    Returns
    -------
    Quadrotor
        Ready-to-fly quadrotor instance.
    """
    if preset == VehiclePreset.CUSTOM:
        return Quadrotor(QuadrotorParams(**overrides))
    if preset not in _QUAD_PRESETS:
        raise ValueError(
            f"{preset.value!r} is not a quadrotor "
            f"({_PRESETS[preset].n_rotors} rotors). Use create_multirotor()."
        )
    params = _apply_overrides(_QUAD_PRESETS[preset], overrides)
    return Quadrotor(params)  # type: ignore[arg-type]


def create_multirotor(
    preset: VehiclePreset = VehiclePreset.HEX_S550,
    **overrides: object,
) -> Multirotor:
    """Create a :class:`Multirotor` from any named preset.

    Works for every preset, four-rotor ones included — a
    :class:`Quadrotor` is returned for those, so the aircraft keeps the
    quadrotor-specific parameters (``arm_length``, ``frame``) that the rest
    of the library reaches for.

    Parameters
    ----------
    preset : which platform to use.
    **overrides : per-field overrides, forwarded to whichever params class
        the preset uses. ``rotors=`` replaces the layout outright.

    Returns
    -------
    Multirotor
        Ready-to-fly aircraft.
    """
    if preset == VehiclePreset.CUSTOM:
        return Multirotor(MultirotorParams(**overrides))  # type: ignore[arg-type]
    if preset in _QUAD_PRESETS:
        return create_quadrotor(preset, **overrides)  # type: ignore[arg-type]
    params = _apply_overrides(_MULTI_PRESETS[preset], overrides)
    return Multirotor(params)  # type: ignore[arg-type]


def get_params(preset: VehiclePreset) -> QuadrotorParams | MultirotorParams:
    """Return the parameters for a named preset."""
    return _PRESETS[preset]
