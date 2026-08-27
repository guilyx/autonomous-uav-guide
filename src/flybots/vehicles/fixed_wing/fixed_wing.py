# Erwin Lejeune - 2026-02-17
"""Six-degree-of-freedom fixed-wing aircraft model.

Reference: R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
Practice*, Princeton University Press, 2012, Chapters 3-4 and Appendix E.

Frame conventions (shared with every other vehicle in this library):

* World is **ENU** — ``z`` is altitude, increasing upward.
* Body is **FLU** — ``x`` forward, ``y`` left, ``z`` up.
* Euler angles are ZYX ``(roll, pitch, yaw)``, and ``rotation_matrix``
  maps body → world.

Because a Forward-Left-Up frame flips ``y`` and ``z`` relative to the
Forward-Right-Down frame that aerodynamics textbooks use, **a positive
pitch angle is nose-down** here. Use :attr:`FixedWing.pitch_up` if you want
the aerospace sign convention for display purposes.

The aerodynamics themselves live in :mod:`~flybots.vehicles.fixed_wing.
aerodynamics` and are evaluated in the textbook FRD frame; this class
handles the conversion in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from flybots.frames.transforms import (
    euler_rates_from_body_rates,
    euler_to_rotation,
    flu_to_frd,
    frd_to_flu,
    gravity_in_body,
)
from flybots.vehicles.base import UAVBase, UAVParams
from flybots.vehicles.fixed_wing.aerodynamics import (
    AeroCoefficients,
    AeroWrench,
    PropulsionParams,
    aero_wrench,
    flow_state,
)

__all__ = ["FixedWing", "FixedWingParams"]

# Body rates beyond this are non-physical for an airframe and only ever
# arise from a diverging integration; clamp instead of producing NaN.
_RATE_LIMIT = 20.0
_SURFACE_LIMIT = np.radians(30.0)


@dataclass
class FixedWingParams(UAVParams):
    """Physical parameters for a fixed-wing aircraft.

    Defaults describe the Aerosonde UAV used as the running example in
    Beard & McLain (13.5 kg, 2.9 m span).
    """

    # ── geometry ──────────────────────────────────────────────────────────
    wing_area: float = 0.55
    wing_span: float = 2.8956
    chord: float = 0.18994
    rho_air: float = 1.2682

    mass: float = 13.5
    inertia: NDArray[np.floating] = field(default_factory=lambda: np.diag([0.8244, 1.135, 1.759]))

    # ── aerodynamics and propulsion ───────────────────────────────────────
    coeffs: AeroCoefficients = field(default_factory=AeroCoefficients)
    propulsion: PropulsionParams = field(default_factory=PropulsionParams)

    # ── operating envelope (used by presets, trim and the autopilot) ──────
    cruise_airspeed: float = 35.0
    """Nominal cruise airspeed [m/s]."""

    @property
    def max_lift_coefficient(self) -> float:
        """Peak ``CL`` of the stall model, found by sweeping alpha."""
        from flybots.vehicles.fixed_wing.aerodynamics import lift_coefficient

        grid = np.linspace(0.0, np.pi / 2, 181)
        return max(lift_coefficient(float(a), self.coeffs) for a in grid)

    @property
    def stall_airspeed(self) -> float:
        """1-g stall speed [m/s], derived from the model rather than assumed.

        ``V_stall = sqrt(2 m g / (rho S CL_max))`` — the speed below which
        the wing cannot generate enough lift to hold the aircraft up.
        """
        denominator = self.rho_air * self.wing_area * self.max_lift_coefficient
        return float(np.sqrt(2.0 * self.mass * self.gravity / max(denominator, 1e-9)))

    # ── backwards-compatible aliases ──────────────────────────────────────
    # These used to be top-level fields. They now forward to ``coeffs`` so
    # existing code keeps working and the two can never drift apart.

    @property
    def CL0(self) -> float:
        return self.coeffs.CL0

    @property
    def CLa(self) -> float:
        return self.coeffs.CLa

    @property
    def CD0(self) -> float:
        return self.coeffs.CD0

    @property
    def CDa(self) -> float:
        return self.coeffs.CDa

    @property
    def Cm0(self) -> float:
        return self.coeffs.Cm0

    @property
    def Cma(self) -> float:
        return self.coeffs.Cma

    @property
    def e_oswald(self) -> float:
        return self.coeffs.e_oswald

    @property
    def aspect_ratio(self) -> float:
        """Wing aspect ratio ``b^2 / S``."""
        return self.wing_span**2 / self.wing_area

    def wing_loading(self) -> float:
        """Wing loading [N/m^2] — mass supported per unit wing area."""
        return self.mass * self.gravity / self.wing_area


class FixedWing(UAVBase):
    """Non-linear 6DOF fixed-wing aircraft.

    State (12 elements)::

        [x, y, z, phi, theta, psi, u, v, w, p, q, r]

    * ``x, y, z`` — world ENU position [m], ``z`` is altitude.
    * ``phi, theta, psi`` — ZYX Euler angles [rad].
    * ``u, v, w`` — **body-frame** velocity [m/s] in Forward-Left-Up.
    * ``p, q, r`` — body angular rates [rad/s].

    Control (4 elements)::

        [elevator, aileron, rudder, throttle]

    Surface deflections are in radians and clamped to +-30 deg; throttle is
    clamped to ``[0, 1]``.

    Examples
    --------
    >>> from flybots.vehicles.fixed_wing import FixedWing
    >>> aircraft = FixedWing()
    >>> aircraft.reset_trimmed(airspeed=35.0, altitude=100.0)
    >>> aircraft.airspeed  # doctest: +ELLIPSIS
    35.0...
    """

    STATE_SIZE = 12

    IX, IY, IZ = 0, 1, 2
    IPHI, ITHETA, IPSI = 3, 4, 5
    IU, IV, IW = 6, 7, 8
    IP, IQ, IR = 9, 10, 11

    def __init__(self, params: FixedWingParams | None = None) -> None:
        self.fw_params = params or FixedWingParams()
        super().__init__(self.fw_params)
        self._last_wrench: AeroWrench | None = None

    # ── shape ─────────────────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self.STATE_SIZE

    @property
    def control_dim(self) -> int:
        return 4

    # ── convenience accessors ─────────────────────────────────────────────

    @property
    def euler(self) -> NDArray[np.floating]:
        """``[roll, pitch, yaw]`` [rad]."""
        return self._state[3:6].copy()

    @property
    def body_velocity(self) -> NDArray[np.floating]:
        """``[u, v, w]`` in the FLU body frame [m/s]."""
        return self._state[6:9].copy()

    @property
    def velocity(self) -> NDArray[np.floating]:
        """Velocity in the world ENU frame [m/s]."""
        return self.rotation_matrix() @ self._state[6:9]

    @property
    def angular_velocity(self) -> NDArray[np.floating]:
        return self._state[9:12].copy()

    @property
    def airspeed(self) -> float:
        """True airspeed :math:`V_a` [m/s]."""
        return float(np.linalg.norm(self._state[6:9]))

    @property
    def alpha(self) -> float:
        """Angle of attack [rad], positive nose-up relative to the airflow."""
        return flow_state(flu_to_frd(self._state[6:9]), self.fw_params.rho_air).alpha

    @property
    def beta(self) -> float:
        """Sideslip angle [rad], positive with relative wind from the right."""
        return flow_state(flu_to_frd(self._state[6:9]), self.fw_params.rho_air).beta

    @property
    def pitch_up(self) -> float:
        """Pitch attitude in the aerospace sign convention (positive nose-up).

        The stored ``theta`` is positive nose-*down* because the body frame
        is Forward-Left-Up; this property just negates it for display.
        """
        return -float(self._state[4])

    @property
    def flight_path_angle(self) -> float:
        """Climb angle [rad] — positive means gaining altitude."""
        vel = self.velocity
        horizontal = float(np.hypot(vel[0], vel[1]))
        return float(np.arctan2(vel[2], max(horizontal, 1e-9)))

    @property
    def load_factor(self) -> float:
        """Lift-to-weight ratio (the "g" the airframe is pulling)."""
        if self._last_wrench is None:
            return 1.0
        return self._last_wrench.lift / (self.fw_params.mass * self.fw_params.gravity)

    @property
    def last_wrench(self) -> AeroWrench | None:
        """Aerodynamic wrench from the most recent dynamics evaluation."""
        return self._last_wrench

    def rotation_matrix(self) -> NDArray[np.floating]:
        """Body → world rotation matrix for the current attitude."""
        return euler_to_rotation(*self._state[3:6])

    def is_stalled(self) -> bool:
        """True when the angle of attack exceeds the stall boundary."""
        return abs(self.alpha) > self.fw_params.coeffs.alpha_stall

    # ── setup helpers ─────────────────────────────────────────────────────

    def reset_trimmed(
        self,
        *,
        airspeed: float | None = None,
        altitude: float = 100.0,
        heading: float = 0.0,
        climb_rate: float = 0.0,
    ) -> NDArray[np.floating]:
        """Reset into steady level (or steady-climb) flight.

        Solves for the trim condition first, so the aircraft starts already
        balanced instead of pitching and porpoising for the first seconds of
        a simulation. Returns the trim control vector, which is what you
        want as the feed-forward term for a controller.
        """
        from flybots.vehicles.fixed_wing.trim import compute_trim

        va = airspeed if airspeed is not None else self.fw_params.cruise_airspeed
        trim = compute_trim(self.fw_params, airspeed=va, climb_rate=climb_rate)
        state = np.zeros(self.STATE_SIZE)
        state[2] = altitude
        state[3:6] = [0.0, trim.theta, heading]
        state[6:9] = trim.body_velocity
        self.reset(state=state)
        return trim.controls

    @staticmethod
    def clamp_controls(control: NDArray[np.floating]) -> NDArray[np.floating]:
        """Clamp surfaces to +-30 deg and throttle to ``[0, 1]``."""
        control = np.asarray(control, dtype=float).copy()
        control[:3] = np.clip(control[:3], -_SURFACE_LIMIT, _SURFACE_LIMIT)
        control[3] = np.clip(control[3], 0.0, 1.0)
        return control

    # ── dynamics ──────────────────────────────────────────────────────────

    def _dynamics(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        p = self.fw_params
        phi, theta = state[3], state[4]
        vel_flu = state[6:9]
        rates_flu = state[9:12]

        # Aerodynamics are written in the textbook FRD frame.
        wrench = aero_wrench(
            velocity_body_frd=flu_to_frd(vel_flu),
            rates_body_frd=flu_to_frd(rates_flu),
            controls=control,
            coeffs=p.coeffs,
            prop=p.propulsion,
            wing_area=p.wing_area,
            wing_span=p.wing_span,
            chord=p.chord,
            rho=p.rho_air,
        )
        self._last_wrench = wrench

        force_flu = frd_to_flu(wrench.force)
        moment_flu = frd_to_flu(wrench.moment)

        # Translational: body-frame Newton-Euler with the transport term.
        accel = force_flu / p.mass + gravity_in_body(phi, theta, p.gravity)
        accel -= np.cross(rates_flu, vel_flu)

        # Rotational: I w' = tau - w x (I w).
        inertia = p.inertia
        rate_dot = np.linalg.solve(inertia, moment_flu - np.cross(rates_flu, inertia @ rates_flu))

        derivative = np.empty(self.STATE_SIZE)
        derivative[0:3] = euler_to_rotation(phi, theta, state[5]) @ vel_flu
        derivative[3:6] = euler_rates_from_body_rates(phi, theta, rates_flu)
        derivative[6:9] = accel
        derivative[9:12] = rate_dot
        return derivative

    def step(self, control: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Advance the aircraft by ``dt`` seconds (RK4).

        Returns the new state. Controls are clamped to the physical
        actuator envelope before integration.
        """
        control = self.clamp_controls(control)
        super().step(control, dt)

        # Keep the integration inside the region where the model is valid.
        self._state[9:12] = np.clip(self._state[9:12], -_RATE_LIMIT, _RATE_LIMIT)
        self._state[3:6] = (self._state[3:6] + np.pi) % (2 * np.pi) - np.pi
        if self._state[2] < 0.0:
            self._state[2] = 0.0
        return self._state.copy()
