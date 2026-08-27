# Erwin Lejeune - 2026-02-17
"""Trim solver for fixed-wing aircraft.

Trim is the control input and attitude that make steady flight an
*equilibrium* — the aircraft holds airspeed, altitude and attitude with no
further intervention. Starting a simulation from trim removes the pitch
transient that otherwise dominates the first few seconds, and the trim
controls double as the feed-forward term for an autopilot.

Reference: R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
Practice*, Princeton University Press, 2012, Chapter 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from flybots.vehicles.fixed_wing.aerodynamics import aero_wrench

__all__ = ["TrimPoint", "compute_trim", "TrimError"]


class TrimError(RuntimeError):
    """Raised when no trim solution exists for the requested condition."""


@dataclass(frozen=True)
class TrimPoint:
    """A steady-flight equilibrium.

    Attributes
    ----------
    airspeed
        True airspeed the solution was computed for [m/s].
    alpha
        Trim angle of attack [rad].
    theta
        Trim pitch angle [rad] **in the library's FLU convention**, i.e.
        positive is nose-down. Assign it straight into ``state[4]``.
    elevator, throttle
        Trim control settings.
    controls
        ``[elevator, aileron, rudder, throttle]`` ready to feed to
        :meth:`~flybots.vehicles.fixed_wing.FixedWing.step`.
    body_velocity
        Trim ``[u, v, w]`` in the FLU body frame [m/s].
    residual
        Norm of the equilibrium residual. Small means a genuine trim point.
    """

    airspeed: float
    alpha: float
    theta: float
    elevator: float
    throttle: float
    controls: NDArray[np.floating]
    body_velocity: NDArray[np.floating]
    residual: float

    @property
    def alpha_deg(self) -> float:
        return float(np.degrees(self.alpha))

    @property
    def pitch_up_deg(self) -> float:
        """Trim pitch in the aerospace convention (positive nose-up) [deg]."""
        return float(np.degrees(-self.theta))


def compute_trim(
    params,
    *,
    airspeed: float,
    climb_rate: float = 0.0,
    residual_tolerance: float = 1e-3,
) -> TrimPoint:
    """Solve for steady wings-level flight at ``airspeed``.

    Finds the ``(alpha, elevator, throttle)`` that simultaneously zero the
    body-axis force balance and the pitching moment.

    Parameters
    ----------
    params
        A :class:`~flybots.vehicles.fixed_wing.FixedWingParams`.
    airspeed
        Target true airspeed [m/s].
    climb_rate
        Target vertical speed [m/s]. Zero gives level flight.
    residual_tolerance
        Largest normalised residual still accepted as trimmed.

    Raises
    ------
    TrimError
        If the requested condition is unreachable — typically an airspeed
        below stall, or a climb rate the propeller cannot sustain.
    """
    if airspeed <= 0.0:
        raise TrimError("Trim airspeed must be positive.")
    if abs(climb_rate) > airspeed:
        raise TrimError(
            f"Climb rate {climb_rate} m/s exceeds airspeed {airspeed} m/s — "
            "the flight path angle would be undefined."
        )

    gravity = params.gravity
    mass = params.mass
    gamma = float(np.arcsin(np.clip(climb_rate / airspeed, -1.0, 1.0)))

    def residual(x: NDArray[np.floating]) -> NDArray[np.floating]:
        alpha, elevator, throttle = x
        # Aerospace (nose-up positive) pitch attitude for this flight path.
        theta_aero = alpha + gamma
        # Body velocity in the FRD frame the aero model expects.
        vel_frd = np.array(
            [airspeed * np.cos(alpha), 0.0, airspeed * np.sin(alpha)],
            dtype=float,
        )
        wrench = aero_wrench(
            velocity_body_frd=vel_frd,
            rates_body_frd=np.zeros(3),
            controls=np.array([elevator, 0.0, 0.0, throttle]),
            coeffs=params.coeffs,
            prop=params.propulsion,
            wing_area=params.wing_area,
            wing_span=params.wing_span,
            chord=params.chord,
            rho=params.rho_air,
        )
        fx, _, fz = wrench.force
        pitching_moment = wrench.moment[1]

        # Steady flight: accelerations vanish, so applied force exactly
        # cancels the body-frame gravity components.
        return np.array(
            [
                fx / mass - gravity * np.sin(theta_aero),
                fz / mass + gravity * np.cos(theta_aero),
                pitching_moment / (mass * gravity * params.chord),
            ]
        )

    # Seed from a plausible small positive alpha and mid throttle. Several
    # starts guard against the solver stalling in a flat region of the
    # post-stall lift curve.
    seeds = (
        np.array([0.05, 0.0, 0.5]),
        np.array([0.15, -0.1, 0.7]),
        np.array([0.02, 0.05, 0.3]),
    )
    best_x: np.ndarray | None = None
    best_norm = np.inf
    bounds = (
        np.array([-0.5, -np.radians(30.0), 0.0]),
        np.array([0.5, np.radians(30.0), 1.0]),
    )
    for seed in seeds:
        solution = least_squares(residual, seed, bounds=bounds, xtol=1e-12, ftol=1e-12)
        norm = float(np.linalg.norm(solution.fun))
        if norm < best_norm:
            best_x, best_norm = solution.x, norm
        if best_norm < residual_tolerance:
            break

    if best_x is None or best_norm > residual_tolerance:
        raise TrimError(
            f"No trim found at {airspeed:.1f} m/s with {climb_rate:+.1f} m/s climb "
            f"(residual {best_norm:.3g}). The airspeed is likely below stall "
            f"(~{params.stall_airspeed:.1f} m/s) or the climb rate is too steep."
        )

    alpha, elevator, throttle = best_x
    return TrimPoint(
        airspeed=airspeed,
        alpha=float(alpha),
        # FLU pitch is the negative of the aerospace pitch attitude.
        theta=float(-(alpha + gamma)),
        elevator=float(elevator),
        throttle=float(throttle),
        controls=np.array([elevator, 0.0, 0.0, throttle]),
        # FLU body velocity: w flips sign relative to FRD.
        body_velocity=np.array(
            [airspeed * np.cos(alpha), 0.0, -airspeed * np.sin(alpha)],
            dtype=float,
        ),
        residual=norm,
    )
