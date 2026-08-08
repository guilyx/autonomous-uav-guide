# Erwin Lejeune - 2026-02-17
"""Aerodynamic coefficient model for fixed-wing aircraft.

Implements the full non-linear aerodynamic force and moment model of

    R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
    Practice*, Princeton University Press, 2012 — Chapter 4 and Appendix E.

Everything in this module works in the **standard aerodynamic body frame**
(``x`` forward, ``y`` right, ``z`` down / FRD), which is what the textbook
equations are written in. The vehicle model in :mod:`fixed_wing` converts
between that and the library-wide FLU body frame; see
:func:`uav_sim.frames.transforms.flu_to_frd`.

The coefficient set is split into three groups, matching the textbook:

``longitudinal``
    Lift, drag and pitching moment as functions of angle of attack
    :math:`\\alpha`, pitch rate :math:`q` and elevator :math:`\\delta_e`.
``lateral``
    Side force, roll and yaw moment as functions of sideslip
    :math:`\\beta`, roll/yaw rate and aileron/rudder deflection.
``propulsion``
    Propeller thrust and reaction torque as a function of throttle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AeroCoefficients",
    "AeroState",
    "AeroWrench",
    "PropulsionParams",
    "aero_wrench",
    "airframe_wrench",
    "blending_sigma",
    "drag_coefficient",
    "flow_state",
    "lift_coefficient",
    "propeller_thrust",
]

# Exponent clamp keeping the stall blending function finite for any alpha.
_EXP_CLAMP = 60.0
# Airspeed floor used only inside dimensionless rate ratios.
_VA_FLOOR = 1e-6


@dataclass
class AeroCoefficients:
    """Dimensionless stability and control derivatives.

    Defaults are the Aerosonde UAV coefficients tabulated in Beard &
    McLain, Appendix E. Every field is used by :func:`aero_wrench` — there
    are no decorative parameters.
    """

    # ── longitudinal: lift ────────────────────────────────────────────────
    CL0: float = 0.23
    CLa: float = 5.61
    CLq: float = 7.95
    CLde: float = 0.13

    # ── longitudinal: drag ────────────────────────────────────────────────
    CD0: float = 0.043
    CDa: float = 0.030
    CDq: float = 0.0
    CDde: float = 0.0135
    CDp: float = 0.0
    """Parasitic drag added on top of the induced-drag polar."""

    # ── longitudinal: pitching moment ─────────────────────────────────────
    Cm0: float = 0.0135
    Cma: float = -2.74
    """Static pitch stability. Must be negative for a stable aircraft."""
    Cmq: float = -38.21
    """Pitch damping. Must be negative or the short-period mode diverges."""
    Cmde: float = -0.99

    # ── lateral: side force ───────────────────────────────────────────────
    CY0: float = 0.0
    CYb: float = -0.98
    CYp: float = 0.0
    CYr: float = 0.0
    CYda: float = 0.075
    CYdr: float = 0.19

    # ── lateral: rolling moment ───────────────────────────────────────────
    Cl0: float = 0.0
    Clb: float = -0.13
    """Dihedral effect. Negative gives roll-into-sideslip stability."""
    Clp: float = -0.51
    Clr: float = 0.25
    Clda: float = 0.17
    Cldr: float = 0.0024

    # ── lateral: yawing moment ────────────────────────────────────────────
    Cn0: float = 0.0
    Cnb: float = 0.073
    """Weathercock stability. Positive keeps the nose into the wind."""
    Cnp: float = -0.069
    Cnr: float = -0.095
    """Yaw damping."""
    Cnda: float = -0.011
    Cndr: float = -0.069

    # ── stall model ───────────────────────────────────────────────────────
    alpha_stall: float = 0.47
    """Stall angle of attack [rad] (~27 deg for the Aerosonde)."""
    stall_sharpness: float = 50.0
    """``M`` in the sigmoid blend between linear lift and flat-plate lift."""

    e_oswald: float = 0.9
    """Oswald span efficiency, drives induced drag."""


@dataclass
class PropulsionParams:
    """Simplified propeller model (Beard & McLain, eq. 4.18).

    Thrust falls off as the aircraft speeds up, which is what makes level
    flight settle at a finite airspeed for a given throttle.
    """

    prop_area: float = 0.2027
    """Swept propeller disc area ``S_prop`` [m^2]."""
    prop_efficiency: float = 1.0
    """``C_prop`` — momentum-theory efficiency factor."""
    k_motor: float = 80.0
    """Motor constant: exit airspeed at full throttle [m/s]."""
    k_torque: float = 0.0
    """``k_{T_P}`` — propeller reaction torque constant."""
    k_omega: float = 0.0
    """``k_\\Omega`` — propeller speed constant [rad/s per unit throttle]."""


@dataclass(frozen=True)
class AeroState:
    """Air-relative flow condition used to evaluate the coefficients."""

    airspeed: float
    """True airspeed :math:`V_a` [m/s]."""
    alpha: float
    """Angle of attack [rad], positive nose-up relative to the airflow."""
    beta: float
    """Sideslip angle [rad], positive with the relative wind from the right."""
    dynamic_pressure: float
    """:math:`\\bar q = \\tfrac12 \\rho V_a^2` [Pa]."""


@dataclass(frozen=True)
class AeroWrench:
    """Aerodynamic + propulsive force and moment in the FRD body frame."""

    force: NDArray[np.floating]
    """``[fx, fy, fz]`` [N]."""
    moment: NDArray[np.floating]
    """``[l, m, n]`` [N.m]."""
    lift: float
    """Lift magnitude [N], reported for plotting and diagnostics."""
    drag: float
    """Drag magnitude [N]."""
    thrust: float
    """Propeller thrust [N]."""


def blending_sigma(alpha: float, alpha_stall: float, sharpness: float) -> float:
    """Sigmoid blend between attached and stalled flow.

    Returns 0 well inside the linear regime and 1 well past stall
    (Beard & McLain, eq. 4.10). Exponents are clamped so the function stays
    finite for tumbling angles of attack.
    """
    e_neg = np.exp(np.clip(-sharpness * (alpha - alpha_stall), -_EXP_CLAMP, _EXP_CLAMP))
    e_pos = np.exp(np.clip(sharpness * (alpha + alpha_stall), -_EXP_CLAMP, _EXP_CLAMP))
    return float((1.0 + e_neg + e_pos) / ((1.0 + e_neg) * (1.0 + e_pos)))


def lift_coefficient(alpha: float, c: AeroCoefficients) -> float:
    """Lift coefficient with post-stall flat-plate behaviour.

    Below :attr:`AeroCoefficients.alpha_stall` this is the linear
    ``CL0 + CLa * alpha``. Past stall it blends into the flat-plate result
    ``2 sign(a) sin^2(a) cos(a)``, so lift drops instead of growing without
    bound.
    """
    sigma = blending_sigma(alpha, c.alpha_stall, c.stall_sharpness)
    linear = c.CL0 + c.CLa * alpha
    flat_plate = 2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    return float((1.0 - sigma) * linear + sigma * flat_plate)


def drag_coefficient(alpha: float, c: AeroCoefficients, aspect_ratio: float) -> float:
    """Drag coefficient from the induced-drag polar (Beard & McLain, eq. 4.11).

    ``CD = CDp + (CL0 + CLa a)^2 / (pi e AR)`` — this is where
    :attr:`AeroCoefficients.e_oswald` and the wing span actually enter the
    model, via the aspect ratio.
    """
    cl_linear = c.CL0 + c.CLa * alpha
    induced = cl_linear**2 / (np.pi * c.e_oswald * max(aspect_ratio, 1e-6))
    return float(c.CD0 + c.CDp + induced + c.CDa * alpha**2)


def flow_state(
    velocity_body_frd: NDArray[np.floating],
    rho: float,
) -> AeroState:
    """Convert an FRD body velocity into airspeed, alpha and beta."""
    u, v, w = velocity_body_frd
    airspeed = float(np.sqrt(u * u + v * v + w * w))
    alpha = float(np.arctan2(w, u)) if abs(u) + abs(w) > 1e-12 else 0.0
    beta = float(np.arcsin(np.clip(v / airspeed, -1.0, 1.0))) if airspeed > 1e-9 else 0.0
    return AeroState(
        airspeed=airspeed,
        alpha=alpha,
        beta=beta,
        dynamic_pressure=0.5 * rho * airspeed**2,
    )


def propeller_thrust(
    throttle: float,
    airspeed: float,
    rho: float,
    prop: PropulsionParams,
) -> tuple[float, float]:
    """Return ``(thrust [N], reaction torque [N.m])`` for a throttle setting.

    Momentum theory: the propeller accelerates air from :math:`V_a` to
    :math:`k_{motor}\\,\\delta_t`, so thrust vanishes once the aircraft
    reaches the propeller exit speed.
    """
    exit_speed = prop.k_motor * throttle
    thrust = 0.5 * rho * prop.prop_area * prop.prop_efficiency * (exit_speed**2 - airspeed**2)
    torque = -prop.k_torque * (prop.k_omega * throttle) ** 2
    return float(thrust), float(torque)


def airframe_wrench(
    *,
    velocity_body_frd: NDArray[np.floating],
    rates_body_frd: NDArray[np.floating],
    surfaces: NDArray[np.floating],
    coeffs: AeroCoefficients,
    wing_area: float,
    wing_span: float,
    chord: float,
    rho: float,
) -> AeroWrench:
    """Purely aerodynamic wrench — wing, tail and control surfaces, no thrust.

    Split out from :func:`aero_wrench` so airframes that generate thrust
    some other way (a tiltrotor, for instance) can reuse the same wing
    model without inheriting a propeller.

    Parameters
    ----------
    velocity_body_frd
        Air-relative velocity ``[u, v, w]`` [m/s].
    rates_body_frd
        Body angular rates ``[p, q, r]`` [rad/s].
    surfaces
        ``[elevator, aileron, rudder]`` deflections [rad].
    coeffs
        Coefficient set.
    wing_area, wing_span, chord, rho
        Geometry and air density.
    """
    elevator, aileron, rudder = surfaces
    p_rate, q_rate, r_rate = rates_body_frd

    flow = flow_state(velocity_body_frd, rho)
    va = max(flow.airspeed, _VA_FLOOR)
    qbar_s = flow.dynamic_pressure * wing_area

    # Dimensionless rate ratios. qbar_s is O(Va^2) and these are O(1/Va),
    # so every product below stays finite as the aircraft slows to a stop.
    c_2va = chord / (2.0 * va)
    b_2va = wing_span / (2.0 * va)
    aspect_ratio = wing_span**2 / wing_area

    # ── longitudinal ──────────────────────────────────────────────────────
    cl = lift_coefficient(flow.alpha, coeffs) + coeffs.CLq * c_2va * q_rate
    cl += coeffs.CLde * elevator
    cd = drag_coefficient(flow.alpha, coeffs, aspect_ratio) + coeffs.CDq * c_2va * q_rate
    cd += coeffs.CDde * abs(elevator)

    lift = qbar_s * cl
    drag = qbar_s * cd

    ca, sa = np.cos(flow.alpha), np.sin(flow.alpha)
    # Rotate stability-axis lift/drag into body axes (Beard eq. 4.6).
    fx = -drag * ca + lift * sa
    fz = -drag * sa - lift * ca

    pitch_moment = (
        qbar_s
        * chord
        * (
            coeffs.Cm0
            + coeffs.Cma * flow.alpha
            + coeffs.Cmq * c_2va * q_rate
            + coeffs.Cmde * elevator
        )
    )

    # ── lateral-directional ───────────────────────────────────────────────
    fy = qbar_s * (
        coeffs.CY0
        + coeffs.CYb * flow.beta
        + coeffs.CYp * b_2va * p_rate
        + coeffs.CYr * b_2va * r_rate
        + coeffs.CYda * aileron
        + coeffs.CYdr * rudder
    )
    roll_moment = (
        qbar_s
        * wing_span
        * (
            coeffs.Cl0
            + coeffs.Clb * flow.beta
            + coeffs.Clp * b_2va * p_rate
            + coeffs.Clr * b_2va * r_rate
            + coeffs.Clda * aileron
            + coeffs.Cldr * rudder
        )
    )
    yaw_moment = (
        qbar_s
        * wing_span
        * (
            coeffs.Cn0
            + coeffs.Cnb * flow.beta
            + coeffs.Cnp * b_2va * p_rate
            + coeffs.Cnr * b_2va * r_rate
            + coeffs.Cnda * aileron
            + coeffs.Cndr * rudder
        )
    )

    return AeroWrench(
        force=np.array([fx, fy, fz]),
        moment=np.array([roll_moment, pitch_moment, yaw_moment]),
        lift=float(lift),
        drag=float(drag),
        thrust=0.0,
    )


def aero_wrench(
    *,
    velocity_body_frd: NDArray[np.floating],
    rates_body_frd: NDArray[np.floating],
    controls: NDArray[np.floating],
    coeffs: AeroCoefficients,
    prop: PropulsionParams,
    wing_area: float,
    wing_span: float,
    chord: float,
    rho: float,
) -> AeroWrench:
    """Total aerodynamic **and** propulsive wrench in the FRD body frame.

    Parameters
    ----------
    velocity_body_frd
        Air-relative velocity ``[u, v, w]`` [m/s].
    rates_body_frd
        Body angular rates ``[p, q, r]`` [rad/s].
    controls
        ``[elevator, aileron, rudder, throttle]``. Surfaces in radians,
        throttle in ``[0, 1]``.
    coeffs, prop
        Coefficient and propulsion parameter sets.
    wing_area, wing_span, chord, rho
        Geometry and air density.
    """
    airframe = airframe_wrench(
        velocity_body_frd=velocity_body_frd,
        rates_body_frd=rates_body_frd,
        surfaces=np.asarray(controls)[:3],
        coeffs=coeffs,
        wing_area=wing_area,
        wing_span=wing_span,
        chord=chord,
        rho=rho,
    )
    airspeed = float(np.linalg.norm(velocity_body_frd))
    thrust, prop_torque = propeller_thrust(float(controls[3]), airspeed, rho, prop)

    return AeroWrench(
        force=airframe.force + np.array([thrust, 0.0, 0.0]),
        moment=airframe.moment + np.array([prop_torque, 0.0, 0.0]),
        lift=airframe.lift,
        drag=airframe.drag,
        thrust=thrust,
    )
