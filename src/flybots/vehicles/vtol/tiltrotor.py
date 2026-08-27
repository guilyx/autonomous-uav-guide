# Erwin Lejeune - 2026-02-17
"""Tilt-rotor VTOL model with physically consistent transition dynamics.

A tilt-rotor is a wing and a set of rotors whose thrust axis rotates from
vertical (hover) to horizontal (cruise). The interesting part is the
*transition*: as the rotors tilt forward the aircraft accelerates, and the
wing progressively takes over lift duty from the rotors.

Reference: R. Bapst et al., "Design and Implementation of an Unmanned
Tail-Sitter," IROS, 2015, adapted for a generic tilt-rotor. The wing model
is the Beard & McLain airframe shared with
:mod:`flybots.vehicles.fixed_wing`.

Frame conventions match the rest of the library: world **ENU**, body
**FLU**, ZYX Euler angles, and positive pitch is nose-down.

Modelling notes
---------------
The wing generates lift whenever there is airflow over it — that is what
makes a transition work, and it depends on **airspeed**, not on where the
rotors happen to be pointing. Angle of attack is measured from the
*body-relative* airflow, so pitch attitude changes the wing's incidence the
way it physically must, and lift acts perpendicular to the relative wind in
the body's plane of symmetry, so banking tilts the lift vector and the
aircraft turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from flybots.frames.transforms import (
    euler_rates_from_body_rates,
    euler_to_rotation,
    flu_to_frd,
    frd_to_flu,
)
from flybots.vehicles.base import UAVBase, UAVParams
from flybots.vehicles.fixed_wing.aerodynamics import (
    AeroCoefficients,
    AeroWrench,
    airframe_wrench,
    flow_state,
)

__all__ = ["Tiltrotor", "TiltrotorParams", "HOVER_TILT", "CRUISE_TILT"]

HOVER_TILT = 0.0
"""Rotor tilt for hover — thrust straight up along body ``+z``."""

CRUISE_TILT = np.pi / 2
"""Rotor tilt for cruise — thrust forward along body ``+x``."""

_RATE_LIMIT = 20.0


def _vtol_wing_coefficients() -> AeroCoefficients:
    """Wing coefficients for a small VTOL airframe.

    A modest-aspect-ratio wing with a stabiliser: stable in pitch, mild
    dihedral effect, and an early stall because VTOL wings are usually
    thick and lightly loaded.
    """
    return replace(
        AeroCoefficients(),
        CL0=0.15,
        CLa=4.5,
        CLq=4.0,
        CLde=0.20,
        CD0=0.055,
        CDa=0.04,
        Cm0=0.01,
        Cma=-1.2,
        Cmq=-12.0,
        Cmde=-0.6,
        CYb=-0.55,
        Clb=-0.08,
        Clp=-0.40,
        Clda=0.12,
        Cnb=0.055,
        Cnr=-0.06,
        Cndr=-0.04,
        alpha_stall=0.32,
        stall_sharpness=30.0,
        e_oswald=0.85,
    )


@dataclass
class TiltrotorParams(UAVParams):
    """Physical parameters for a tilt-rotor VTOL."""

    mass: float = 5.0
    inertia: NDArray[np.floating] = field(default_factory=lambda: np.diag([0.1, 0.1, 0.15]))
    num_rotors: int = 4
    arm_length: float = 0.3

    # ── wing ──────────────────────────────────────────────────────────────
    wing_area: float = 0.4
    wing_span: float = 1.6
    chord: float = 0.25
    rho_air: float = 1.225
    coeffs: AeroCoefficients = field(default_factory=_vtol_wing_coefficients)

    # ── rotors ────────────────────────────────────────────────────────────
    max_tilt: float = CRUISE_TILT
    """Maximum rotor tilt [rad]. ``0`` is hover, ``pi/2`` is cruise."""
    max_thrust_ratio: float = 2.5
    """Peak rotor thrust as a multiple of aircraft weight."""
    tilt_rate_limit: float = np.radians(15.0)
    """How fast the tilt mechanism can move [rad/s]. Actuators are not instant."""

    # ── legacy aliases ────────────────────────────────────────────────────
    # ``CL_alpha`` and ``CD0`` were top-level fields before the wing model
    # was shared with the fixed-wing airframe; they now forward to ``coeffs``.

    @property
    def CL_alpha(self) -> float:
        return self.coeffs.CLa

    @property
    def CD0(self) -> float:
        return self.coeffs.CD0

    @property
    def max_thrust(self) -> float:
        """Peak total rotor thrust [N]."""
        return self.max_thrust_ratio * self.mass * self.gravity

    @property
    def max_torque(self) -> float:
        """Peak control torque [N.m] available from differential rotor thrust.

        Half the rotors push while half pull, each acting through the arm,
        so the achievable couple is bounded by ``arm_length * max_thrust / 2``.
        """
        return 0.5 * self.arm_length * self.max_thrust

    @property
    def stall_airspeed(self) -> float:
        """Wing-borne stall speed [m/s] — the floor for cruise flight."""
        from flybots.vehicles.fixed_wing.aerodynamics import lift_coefficient

        cl_max = max(
            lift_coefficient(float(a), self.coeffs) for a in np.linspace(0.0, np.pi / 2, 181)
        )
        return float(
            np.sqrt(2.0 * self.mass * self.gravity / (self.rho_air * self.wing_area * cl_max))
        )


class Tiltrotor(UAVBase):
    """Tilt-rotor VTOL with a hover-to-cruise transition.

    State (12 elements)::

        [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

    Position and velocity are in the world ENU frame (matching
    :class:`~flybots.vehicles.multirotor.quadrotor.Quadrotor`); angular
    rates are body-frame.

    Control (5 elements)::

        [thrust, tau_x, tau_y, tau_z, tilt]

    ``thrust`` is total rotor thrust [N] along the tilted rotor axis;
    ``tau_*`` are body-frame control torques [N.m] from differential rotor
    thrust and control surfaces; ``tilt`` is the commanded rotor angle
    [rad], slew-limited by :attr:`TiltrotorParams.tilt_rate_limit`.

    Examples
    --------
    >>> import numpy as np
    >>> from flybots.vehicles.vtol import Tiltrotor
    >>> vtol = Tiltrotor()
    >>> _ = vtol.reset(state=np.array([0, 0, 20.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    >>> hover = vtol.vtol_params.mass * vtol.vtol_params.gravity
    >>> for _ in range(100):
    ...     vtol.step(np.array([hover, 0, 0, 0, 0.0]), 0.01)
    >>> bool(abs(vtol.state[2] - 20.0) < 0.1)
    True
    """

    STATE_SIZE = 12

    IX, IY, IZ = 0, 1, 2
    IPHI, ITHETA, IPSI = 3, 4, 5
    IVX, IVY, IVZ = 6, 7, 8
    IP, IQ, IR = 9, 10, 11

    def __init__(self, params: TiltrotorParams | None = None) -> None:
        self.vtol_params = params or TiltrotorParams()
        super().__init__(self.vtol_params)
        self._tilt = HOVER_TILT
        self._last_wrench: AeroWrench | None = None

    @property
    def state_dim(self) -> int:
        return self.STATE_SIZE

    @property
    def control_dim(self) -> int:
        return 5

    # ── accessors ─────────────────────────────────────────────────────────

    @property
    def tilt(self) -> float:
        """Current rotor tilt [rad] after slew limiting."""
        return self._tilt

    @property
    def euler(self) -> NDArray[np.floating]:
        return self._state[3:6].copy()

    @property
    def velocity(self) -> NDArray[np.floating]:
        """World ENU velocity [m/s]."""
        return self._state[6:9].copy()

    @property
    def body_velocity(self) -> NDArray[np.floating]:
        """Air-relative velocity in the FLU body frame [m/s]."""
        return euler_to_rotation(*self._state[3:6]).T @ self._state[6:9]

    @property
    def airspeed(self) -> float:
        return float(np.linalg.norm(self._state[6:9]))

    @property
    def alpha(self) -> float:
        """Angle of attack [rad], measured from the body-relative airflow."""
        return flow_state(flu_to_frd(self.body_velocity), self.vtol_params.rho_air).alpha

    @property
    def beta(self) -> float:
        """Sideslip angle [rad]."""
        return flow_state(flu_to_frd(self.body_velocity), self.vtol_params.rho_air).beta

    @property
    def wing_lift(self) -> float:
        """Lift the wing produced on the last dynamics evaluation [N]."""
        return 0.0 if self._last_wrench is None else self._last_wrench.lift

    @property
    def lift_fraction(self) -> float:
        """Share of the aircraft's weight carried by the wing rather than the rotors.

        Runs from ~0 in hover to ~1 in wing-borne cruise, which is the
        single most useful number for watching a transition.
        """
        weight = self.vtol_params.mass * self.vtol_params.gravity
        return float(np.clip(self.wing_lift / weight, 0.0, 2.0))

    def reset(
        self,
        state: NDArray[np.floating] | None = None,
        time: float = 0.0,
        tilt: float = HOVER_TILT,
    ) -> None:
        """Reset the vehicle, including the tilt actuator position."""
        super().reset(state=state, time=time)
        self._tilt = float(np.clip(tilt, 0.0, self.vtol_params.max_tilt))
        self._last_wrench = None

    # ── dynamics ──────────────────────────────────────────────────────────

    def _dynamics(
        self,
        state: NDArray[np.floating],
        control: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        p = self.vtol_params
        phi, theta, psi = state[3], state[4], state[5]
        velocity_world = state[6:9]
        rates = state[9:12]

        thrust, tau_x, tau_y, tau_z = control[:4]
        thrust = float(np.clip(thrust, 0.0, p.max_thrust))
        tilt = self._tilt

        rotation = euler_to_rotation(phi, theta, psi)

        # ── rotors ────────────────────────────────────────────────────────
        # Tilt rotates the thrust axis from body +z (hover) to body +x
        # (cruise) in the aircraft's plane of symmetry.
        thrust_body = np.array([thrust * np.sin(tilt), 0.0, thrust * np.cos(tilt)])

        # ── wing ──────────────────────────────────────────────────────────
        # Air-relative velocity in the body frame. Using the *body* velocity
        # is what makes angle of attack respond to pitch attitude, and what
        # makes lift rotate with bank so the aircraft can turn.
        velocity_body = rotation.T @ velocity_world
        aero = airframe_wrench(
            velocity_body_frd=flu_to_frd(velocity_body),
            rates_body_frd=flu_to_frd(rates),
            surfaces=np.zeros(3),
            coeffs=p.coeffs,
            wing_area=p.wing_area,
            wing_span=p.wing_span,
            chord=p.chord,
            rho=p.rho_air,
        )
        self._last_wrench = aero
        aero_force_body = frd_to_flu(aero.force)
        aero_moment_body = frd_to_flu(aero.moment)

        # ── translational dynamics (world frame) ──────────────────────────
        force_world = rotation @ (thrust_body + aero_force_body)
        acceleration = force_world / p.mass + np.array([0.0, 0.0, -p.gravity])

        # ── rotational dynamics (body frame) ──────────────────────────────
        control_moment = np.array([tau_x, tau_y, tau_z])
        total_moment = control_moment + aero_moment_body
        rate_dot = np.linalg.solve(p.inertia, total_moment - np.cross(rates, p.inertia @ rates))

        derivative = np.empty(self.STATE_SIZE)
        derivative[0:3] = velocity_world
        derivative[3:6] = euler_rates_from_body_rates(phi, theta, rates)
        derivative[6:9] = acceleration
        derivative[9:12] = rate_dot
        return derivative

    def step(self, control: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Advance by ``dt`` seconds (RK4), slew-limiting the tilt actuator."""
        control = np.asarray(control, dtype=float)
        p = self.vtol_params

        # Move the tilt actuator toward its command at a finite rate before
        # integrating, so a step command cannot teleport the rotors.
        tilt_cmd = float(np.clip(control[4], 0.0, p.max_tilt))
        max_step = p.tilt_rate_limit * dt
        self._tilt += float(np.clip(tilt_cmd - self._tilt, -max_step, max_step))

        super().step(control, dt)

        self._state[9:12] = np.clip(self._state[9:12], -_RATE_LIMIT, _RATE_LIMIT)
        self._state[3:6] = (self._state[3:6] + np.pi) % (2 * np.pi) - np.pi
        if self._state[2] < 0.0:
            self._state[2] = 0.0
            self._state[8] = max(self._state[8], 0.0)
        return self._state.copy()
