# Erwin Lejeune - 2026-02-17
"""Mode-scheduled controller for tilt-rotor VTOL aircraft.

A VTOL has to fly two aircraft with one control law. In hover the rotors
carry the whole airframe and attitude sets the direction of travel; in
cruise the wing carries it and attitude sets angle of attack. The
transition between them is the hard part, because lift authority migrates
from the rotors to the wing while the aircraft must keep flying.

The controller handles this with an explicit mode machine:

.. code-block:: text

    HOVER ──airspeed builds──▶ TRANSITION ──wing-borne──▶ CRUISE
      ▲                                                     │
      └──────────── BACK_TRANSITION ◀───────decelerate──────┘

and a single altitude law that works across all four modes: compute the
vertical force the aircraft needs, subtract what the wing is already
providing, and ask the rotors for the remainder along whatever axis they
currently point. That one expression degrades gracefully from pure rotor
lift to pure wing lift without any blending hacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from flybots.frames.transforms import euler_to_rotation, flu_to_frd, frd_to_flu
from flybots.vehicles.fixed_wing.aerodynamics import airframe_wrench
from flybots.vehicles.vtol.tiltrotor import HOVER_TILT, TiltrotorParams

__all__ = ["VTOLController", "VTOLMode", "VTOLCommand", "VTOLGains"]


class VTOLMode(Enum):
    """Flight mode of the tilt-rotor."""

    HOVER = "hover"
    TRANSITION = "transition"
    CRUISE = "cruise"
    BACK_TRANSITION = "back_transition"


@dataclass
class VTOLCommand:
    """Setpoints for the controller."""

    altitude: float = 20.0
    """Commanded altitude [m]."""
    cruise: bool = False
    """Request wing-borne cruise. Clearing it commands a back-transition."""
    cruise_airspeed: float = 22.0
    """Target airspeed once in cruise [m/s]."""
    heading: float = 0.0
    """Commanded heading [rad], ENU."""


@dataclass
class VTOLGains:
    """Controller gains and mode-switch thresholds."""

    # Altitude (outer) → vertical acceleration.
    altitude_kp: float = 1.8
    altitude_kd: float = 2.2

    # Attitude (inner) → body torques.
    attitude_kp: float = 9.0
    attitude_kd: float = 4.5
    yaw_kp: float = 3.0
    yaw_kd: float = 2.0

    # Forward speed → pitch attitude while still rotor-borne.
    speed_kp: float = 0.03
    max_pitch: float = np.radians(15.0)
    max_roll: float = np.radians(30.0)

    # Wing-borne laws: pitch owns altitude, thrust owns airspeed.
    cruise_altitude_kp: float = 0.010
    """Altitude error [m] → commanded pitch [rad] in cruise."""
    cruise_altitude_kd: float = 0.020
    cruise_speed_kp: float = 0.10
    """Airspeed error [m/s] → thrust, as a fraction of weight."""
    cruise_thrust_bias: float = 0.18
    """Feed-forward cruise thrust as a fraction of weight (roughly 1 / (L/D))."""
    feedforward_stall_margin: float = 0.75
    """Cap the trim-incidence feed-forward at this fraction of stall alpha."""

    # Mode switching.
    transition_start_airspeed: float = 2.0
    """Airspeed at which the tilt schedule begins [m/s]."""
    stall_margin: float = 1.25
    """Cruise requires this multiple of the wing's stall speed."""
    back_transition_airspeed: float = 8.0
    """Drop below this during a back-transition to declare hover [m/s]."""


class VTOLController:
    """Altitude, airspeed and heading control across the flight envelope.

    Examples
    --------
    >>> import numpy as np
    >>> from flybots.vehicles.vtol import Tiltrotor
    >>> from flybots.control.vtol_controller import VTOLController, VTOLCommand
    >>> vtol = Tiltrotor()
    >>> _ = vtol.reset(state=np.array([0, 0, 20.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    >>> pilot = VTOLController(vtol.vtol_params)
    >>> cmd = VTOLCommand(altitude=20.0, cruise=True)
    >>> for _ in range(8000):
    ...     _ = vtol.step(pilot.compute(vtol.state, vtol.tilt, cmd, 0.005), 0.005)
    >>> pilot.mode is not None
    True
    """

    def __init__(
        self,
        params: TiltrotorParams,
        gains: VTOLGains | None = None,
    ) -> None:
        self.params = params
        self.gains = gains or VTOLGains()
        self.mode = VTOLMode.HOVER
        self._tilt_cmd = HOVER_TILT

    def reset(self) -> None:
        """Return to hover mode with the rotors upright."""
        self.mode = VTOLMode.HOVER
        self._tilt_cmd = HOVER_TILT

    # ── mode machine ──────────────────────────────────────────────────────

    @property
    def cruise_entry_airspeed(self) -> float:
        """Airspeed at which the wing can hold the aircraft up [m/s]."""
        return self.params.stall_airspeed * self.gains.stall_margin

    def _update_mode(self, airspeed: float, command: VTOLCommand) -> None:
        g = self.gains
        if command.cruise:
            if self.mode in (VTOLMode.HOVER, VTOLMode.BACK_TRANSITION):
                self.mode = VTOLMode.TRANSITION
            if self.mode is VTOLMode.TRANSITION and airspeed >= self.cruise_entry_airspeed:
                self.mode = VTOLMode.CRUISE
        else:
            if self.mode in (VTOLMode.CRUISE, VTOLMode.TRANSITION):
                self.mode = VTOLMode.BACK_TRANSITION
            if self.mode is VTOLMode.BACK_TRANSITION and airspeed <= g.back_transition_airspeed:
                self.mode = VTOLMode.HOVER

    def _wing_authority(self, airspeed: float, tilt: float) -> float:
        """How wing-borne the aircraft is, from 0 (hover) to 1 (cruise).

        Gated on *both* airspeed and actual rotor tilt, and never more than
        the slower of the two. Airspeed alone is not enough: the rotors
        slew at a finite rate, so there is a window where the aircraft is
        fast enough for the wing but the rotors are still pointing mostly
        upward. Handing lift duty over on airspeed alone during that window
        leaves nobody holding the aircraft up.
        """
        low = self.gains.transition_start_airspeed
        high = self.cruise_entry_airspeed
        by_speed = (airspeed - low) / max(high - low, 1e-6)
        by_tilt = np.sin(tilt) ** 2
        return float(np.clip(min(by_speed, by_tilt), 0.0, 1.0))

    def _target_speed(self, command: VTOLCommand) -> float:
        """Airspeed the current mode is trying to reach [m/s]."""
        if self.mode is VTOLMode.CRUISE:
            return command.cruise_airspeed
        if self.mode is VTOLMode.TRANSITION:
            return max(command.cruise_airspeed, self.cruise_entry_airspeed)
        # Hover and back-transition both want the aircraft stopped.
        return 0.0

    def _tilt_schedule(self, airspeed: float) -> float:
        """Rotor tilt for the current mode and airspeed.

        During the forward transition the tilt is scheduled on *airspeed*
        rather than on a timer: the rotors only tilt further forward once
        the wing has enough flow to take up the slack. That makes the
        transition self-pacing, and safe if the aircraft is heavy or
        climbing.
        """
        g = self.gains
        if self.mode is VTOLMode.HOVER:
            return HOVER_TILT
        if self.mode is VTOLMode.CRUISE:
            return self.params.max_tilt
        if self.mode is VTOLMode.TRANSITION:
            span = max(self.cruise_entry_airspeed - g.transition_start_airspeed, 1e-6)
            progress = (airspeed - g.transition_start_airspeed) / span
            return float(np.clip(progress, 0.0, 1.0) * self.params.max_tilt)
        # Back-transition: rotors go upright *first*, paced by the tilt
        # actuator's own slew rate, and the aircraft bleeds speed off
        # afterwards on drag alone.
        #
        # Scheduling this on airspeed the way the forward transition does
        # would deadlock: the rotors would wait for the aircraft to slow
        # down, but with the rotors still pointing forward the only way to
        # decelerate is to pitch up, which stalls the wing and drops the
        # aircraft out of the sky.
        return HOVER_TILT

    # ── control ───────────────────────────────────────────────────────────

    def _wing_wrench(
        self,
        state: NDArray[np.floating],
        rotation: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Aerodynamic force (world frame) and moment (body frame) from the wing."""
        p = self.params
        velocity_body = rotation.T @ state[6:9]
        aero = airframe_wrench(
            velocity_body_frd=flu_to_frd(velocity_body),
            rates_body_frd=flu_to_frd(state[9:12]),
            surfaces=np.zeros(3),
            coeffs=p.coeffs,
            wing_area=p.wing_area,
            wing_span=p.wing_span,
            chord=p.chord,
            rho=p.rho_air,
        )
        return rotation @ frd_to_flu(aero.force), frd_to_flu(aero.moment)

    def level_flight_alpha(self, airspeed: float) -> float:
        """Angle of attack the wing needs to carry the aircraft at ``airspeed``.

        Inverting the lift equation gives the trim incidence directly::

            CL_required = 2 m g / (rho Va^2 S)
            alpha       = (CL_required - CL0) / CLa

        Feeding this forward is what lets the cruise pitch loop hold
        altitude. A pure error-driven law commands zero pitch at zero
        error, which is exactly the attitude at which a wing produces too
        little lift to stay up.
        """
        p = self.params
        dynamic_pressure = 0.5 * p.rho_air * airspeed**2 * p.wing_area
        if dynamic_pressure < 1e-6:
            return 0.0
        cl_required = p.mass * p.gravity / dynamic_pressure
        alpha = (cl_required - p.coeffs.CL0) / p.coeffs.CLa
        # Keep a margin below stall. Past the stall boundary more incidence
        # means *less* lift, so a feed-forward that saturates at the stall
        # angle would command exactly the attitude that drops the aircraft.
        safe = self.gains.feedforward_stall_margin * p.coeffs.alpha_stall
        return float(np.clip(alpha, -safe, safe))

    def compute(
        self,
        state: NDArray[np.floating],
        tilt: float,
        command: VTOLCommand,
        dt: float,
    ) -> NDArray[np.floating]:
        """Return ``[thrust, tau_x, tau_y, tau_z, tilt_cmd]``.

        Parameters
        ----------
        state
            Tiltrotor state, 12 elements.
        tilt
            The vehicle's *actual* current tilt [rad]. The thrust solution
            has to use where the rotors really point, not where they were
            commanded, or altitude sags whenever the actuator lags.
        command
            Altitude / cruise / heading setpoints.
        dt
            Control period [s]. Unused by the current laws, accepted so the
            signature matches the other controllers in this library.
        """
        del dt
        p = self.params
        g = self.gains
        phi, theta, psi = state[3], state[4], state[5]
        altitude, vertical_speed = state[2], state[8]
        airspeed = float(np.linalg.norm(state[6:9]))

        self._update_mode(airspeed, command)
        self._tilt_cmd = self._tilt_schedule(airspeed)

        rotation = euler_to_rotation(phi, theta, psi)
        wing_force_world, wing_moment_body = self._wing_wrench(state, rotation)

        # ── control allocation blend ──────────────────────────────────────
        # Rotor-borne and wing-borne flight allocate the two actuators to
        # the two objectives in opposite ways:
        #
        #   hover  — thrust holds altitude, pitch sets forward speed
        #   cruise — pitch holds altitude, thrust sets airspeed
        #
        # Blending on airspeed (and not on attitude, which would close a
        # loop through the thing being controlled) hands authority over
        # smoothly instead of switching discontinuously mid-transition.
        blend = self._wing_authority(airspeed, tilt)
        altitude_error = command.altitude - altitude
        target_speed = self._target_speed(command)
        speed_error = target_speed - airspeed

        # ── thrust ────────────────────────────────────────────────────────
        # Rotor-borne: ask the rotors for whatever vertical force the wing
        # is not already supplying, along whatever axis they point.
        vertical_accel_cmd = g.altitude_kp * altitude_error - g.altitude_kd * vertical_speed
        rotor_vertical_force = p.mass * (p.gravity + vertical_accel_cmd) - wing_force_world[2]
        thrust_axis_world = rotation @ np.array([np.sin(tilt), 0.0, np.cos(tilt)])
        # The floor keeps the division finite as the thrust axis nears horizontal.
        hover_thrust = rotor_vertical_force / max(float(thrust_axis_world[2]), 0.25)

        # Wing-borne: thrust just trims airspeed against drag.
        cruise_thrust = (
            p.mass * p.gravity * (g.cruise_thrust_bias + g.cruise_speed_kp * speed_error)
        )
        # Never let the blend starve the aircraft of lift: whatever the
        # allocation says, the rotors must still supply the vertical force
        # the wing is demonstrably not producing.
        hover_thrust = max(hover_thrust, 0.0)

        thrust = float(
            np.clip((1.0 - blend) * hover_thrust + blend * cruise_thrust, 0.0, p.max_thrust)
        )

        # ── pitch ─────────────────────────────────────────────────────────
        # Remember: FLU pitch is nose-DOWN positive, so climbing and
        # accelerating both want a negative sign somewhere.
        hover_pitch = g.speed_kp * speed_error
        # Trim incidence feed-forward plus altitude correction.
        cruise_pitch = -(
            self.level_flight_alpha(airspeed)
            + g.cruise_altitude_kp * altitude_error
            - g.cruise_altitude_kd * vertical_speed
        )
        pitch_cmd = float(
            np.clip(
                (1.0 - blend) * hover_pitch + blend * cruise_pitch,
                -g.max_pitch,
                g.max_pitch,
            )
        )

        roll_cmd = 0.0

        # ── attitude → torques ────────────────────────────────────────────
        # Feed-forward-cancel the wing's own aerodynamic moment before the
        # PD runs. At cruise speed the wing's static stability (Cma) produces
        # a restoring moment an order of magnitude larger than a hover-sized
        # PD can generate, so without this the aircraft simply cannot be
        # held at the incidence it needs to stay airborne.
        inertia = p.inertia
        roll_rate, pitch_rate, yaw_rate = state[9], state[10], state[11]
        heading_error = float(
            np.arctan2(np.sin(command.heading - psi), np.cos(command.heading - psi))
        )
        feedback = np.array(
            [
                inertia[0, 0] * (g.attitude_kp * (roll_cmd - phi) - g.attitude_kd * roll_rate),
                inertia[1, 1] * (g.attitude_kp * (pitch_cmd - theta) - g.attitude_kd * pitch_rate),
                inertia[2, 2] * (g.yaw_kp * heading_error - g.yaw_kd * yaw_rate),
            ]
        )
        torque = np.clip(feedback - wing_moment_body, -p.max_torque, p.max_torque)

        return np.array([thrust, torque[0], torque[1], torque[2], self._tilt_cmd])
