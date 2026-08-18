# Erwin Lejeune - 2026-02-17
"""Successive-loop-closure autopilot for fixed-wing aircraft.

Reference: R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
Practice*, Princeton University Press, 2012, Chapter 6.

The autopilot is a cascade of nested loops:

.. code-block:: text

    airspeed  ──PI───────────────────────────────▶ throttle
    altitude  ──PI──▶ pitch cmd  ──PD───────────▶ elevator
    course    ──PI──▶(+)▶ roll cmd  ──PD────────▶ aileron
    turn bank ───────▲
    sideslip  ──P────┐
    yaw rate  ──washout──D──────────────────────▶ rudder

Inner loops run on attitude (fast), outer loops on trajectory (slow), which
is what keeps the design stable without a full multivariable synthesis.

The one feed-forward path is ``turn bank``: a guidance layer flying a path
of known curvature can hand the autopilot the coordinated-turn bank that
path needs, leaving the course PI to regulate only the error around it. It
defaults to zero, in which case the cascade is exactly the pure feedback
design it has always been — see
:attr:`AutopilotCommand.roll_feedforward`.

Gains are **computed from the airframe**, not hand-tuned. Each loop's
proportional gain comes from a control-authority budget — full actuator
deflection at a stated tracking error — and the achievable bandwidth then
follows from the aircraft's control derivatives. The same autopilot flies a
0.6 kg foam trainer and a 25 kg cargo UAV without retuning.

The course loop closes on **course** (the direction the aircraft is
travelling), not heading. On an airframe that cannot zero its own sideslip
— a rudderless flying wing, say — closing on heading leaves the nose
hunting around an otherwise correct flight path and never settles.

Sign conventions
----------------
This library uses a Forward-Left-Up body frame with an ENU world, so:

* ``theta > 0`` is **nose-down** — the autopilot works internally in the
  aerospace (nose-up positive) convention and converts at the boundary.
* ``phi > 0`` is **bank right**, same as the aerospace convention.
* Yaw ``psi`` increases **counter-clockwise** (ENU), so banking right
  *decreases* the heading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.frames.transforms import euler_to_rotation, flu_to_frd
from uav_sim.vehicles.fixed_wing.aerodynamics import flow_state
from uav_sim.vehicles.fixed_wing.fixed_wing import FixedWingParams
from uav_sim.vehicles.fixed_wing.trim import TrimError, compute_trim

__all__ = ["FixedWingAutopilot", "AutopilotGains", "AutopilotCommand"]

# Bank authority the course PI keeps for itself even when a feed-forward
# turn command has claimed the whole envelope. Without a floor, a
# feed-forward at the roll limit would leave the PI with a zero output
# limit, which silently disables the loop that corrects the error.
_MIN_PI_AUTHORITY = np.radians(1.0)


@dataclass
class AutopilotGains:
    """Loop shaping targets and limits.

    Rather than raw gains, these are the *design* quantities: bandwidths,
    damping ratios and envelope limits. The actual PID gains are derived
    from them and the airframe in :meth:`FixedWingAutopilot._design_gains`.
    """

    # Inner attitude loops.
    #
    # Bandwidth is *derived*, not requested: the gain is set by how much
    # surface deflection you are willing to spend on a given attitude error,
    # and the resulting natural frequency follows. Requesting a bandwidth
    # directly is what produces sign-flipped gains on airframes whose
    # natural dynamics are already faster than the request.
    max_roll_error: float = np.radians(15.0)
    """Roll error that saturates the ailerons. Smaller means tighter."""
    roll_damping: float = 0.9
    max_pitch_error: float = np.radians(15.0)
    """Pitch error that saturates the elevator."""
    pitch_damping: float = 0.9

    # Outer loops, designed the same way: how much inner-loop command are
    # you willing to spend on a given tracking error. Their bandwidth then
    # comes out one to two orders below the inner loops automatically,
    # which is the separation successive loop closure needs.
    max_course_error: float = np.radians(60.0)
    """Course error that commands full bank."""
    course_damping: float = 1.0
    max_altitude_error: float = 30.0
    """Altitude error [m] that commands full pitch."""
    altitude_damping: float = 1.0

    # Airspeed hold via throttle.
    airspeed_kp: float = 0.08
    airspeed_ki: float = 0.03

    # Sideslip regulation and Dutch-roll damping via rudder.
    sideslip_kp: float = 1.5
    yaw_damper_kd: float = 0.8
    """Yaw-rate feedback gain for the Dutch-roll damper."""
    yaw_washout_tau: float = 4.0
    """Washout time constant [s]. Long enough to pass the Dutch-roll
    oscillation and block the steady yaw rate of a turn."""

    # Envelope limits.
    max_roll: float = np.radians(45.0)
    max_pitch: float = np.radians(25.0)
    max_surface: float = np.radians(30.0)


@dataclass
class AutopilotCommand:
    """Setpoints the autopilot tracks."""

    altitude: float = 100.0
    """Commanded altitude [m]."""
    airspeed: float = 35.0
    """Commanded true airspeed [m/s]."""
    course: float = 0.0
    """Commanded heading [rad], ENU convention (counter-clockwise from +x)."""
    roll_feedforward: float = 0.0
    """Bank angle [rad] the commanded flight path is known to require.

    Zero — the default — leaves the autopilot exactly as it was: the course
    PI supplies the whole bank command. A guidance layer flying a path of
    known curvature can instead hand over the coordinated-turn bank
    ``-lambda atan(Va^2 / (g R))`` for that path (negative is left bank in
    this library's ENU frame, so a counter-clockwise turn wants a negative
    value), and the PI is left regulating only the error around it.

    This matters because the course loop's plant is an integrator: entering
    a turn is a *ramp* in commanded course, and a PI tracks a ramp with zero
    steady-state error only after its integrator has charged. On a tight
    circuit the aircraft is never in that steady state, and the lag shows up
    as a bulge on every turn entry and an overshoot on every exit. Beard &
    McLain add the same term for the same reason when following an orbit.

    Measured on a mini-trainer racetrack of 2.5 turn radii, feeding it
    forward takes the worst-case path error from 18.3 m to 6.3 m and the
    steady end-of-leg error from 15.0 m to 2.4 m.
    """


@dataclass
class AutopilotDiagnostics:
    """Inner-loop setpoints, exposed so demos can plot the cascade."""

    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    altitude_error: float = 0.0
    airspeed_error: float = 0.0
    course_error: float = 0.0
    saturated: bool = False


class FixedWingAutopilot:
    """Altitude / airspeed / course hold autopilot.

    Examples
    --------
    >>> from uav_sim.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset
    >>> from uav_sim.control.fixed_wing_autopilot import (
    ...     FixedWingAutopilot, AutopilotCommand)
    >>> aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)
    >>> aircraft.reset_trimmed(altitude=100.0)  # doctest: +ELLIPSIS
    array(...)
    >>> pilot = FixedWingAutopilot(aircraft.fw_params)
    >>> cmd = AutopilotCommand(altitude=120.0, airspeed=18.0, course=0.5)
    >>> for _ in range(2000):
    ...     u = pilot.compute(aircraft.state, cmd, dt=0.01)
    ...     _ = aircraft.step(u, 0.01)
    >>> bool(abs(aircraft.state[2] - 120.0) < 5.0)
    True
    """

    def __init__(
        self,
        params: FixedWingParams,
        gains: AutopilotGains | None = None,
    ) -> None:
        self.params = params
        self.gains = gains or AutopilotGains()
        self.diagnostics = AutopilotDiagnostics()

        self._design_gains()
        self._trim_throttle = self._solve_trim_throttle()
        self.reset()

    # ── setup ─────────────────────────────────────────────────────────────

    def _solve_trim_throttle(self) -> float:
        """Feed-forward throttle for level cruise; 0.5 if the airframe won't trim."""
        try:
            return compute_trim(self.params, airspeed=self.params.cruise_airspeed).throttle
        except TrimError:
            return 0.5

    @staticmethod
    def _damping_gain(target_damping: float, natural_damping: float, authority: float) -> float:
        """Rate-feedback gain that tops the airframe's damping up to the target.

        Solves ``natural + authority * kd = target`` for ``kd``, then floors
        the result at zero *in the stabilising direction*.

        The floor matters. On a stiff airframe the natural damping already
        exceeds the requested one, and the unfloored solution comes out with
        the sign that actively cancels it. That is stable on paper — the
        closed-loop damping still lands on target — but it means feeding
        rate measurements back positively, which amplifies gyro noise for no
        benefit. Being better damped than asked is not a problem worth
        creating a noise amplifier to fix.
        """
        ideal = (target_damping - natural_damping) / authority
        # kd shares the sign of the control authority when it is doing work.
        return float(ideal if ideal * authority > 0.0 else 0.0)

    def _design_gains(self) -> None:
        """Derive PID gains from the airframe's control derivatives.

        Follows Beard & McLain's control-authority design (eqs. 6.5-6.11).
        The proportional gain is fixed by the actuator budget — full
        deflection at :attr:`AutopilotGains.max_roll_error` of error — and
        the achievable natural frequency then *follows* from the aircraft's
        control effectiveness. The derivative gain is whatever brings the
        closed loop to the requested damping ratio.

        Deriving the bandwidth rather than requesting it matters: a
        requested bandwidth below the airframe's natural short-period or
        roll-subsidence frequency asks the controller to *remove* stiffness
        and damping the aircraft already has, which flips the gain signs
        and destabilises the loop.
        """
        p = self.params
        g = self.gains
        c = p.coeffs
        va = p.cruise_airspeed
        qbar_s = 0.5 * p.rho_air * va**2 * p.wing_area
        jx, jy = p.inertia[0, 0], p.inertia[1, 1]

        # ── roll loop: phi'' = -a_phi1 phi' + a_phi2 delta_a ───────────────
        a_phi1 = -qbar_s * p.wing_span * c.Clp * p.wing_span / (2.0 * va) / jx
        a_phi2 = qbar_s * p.wing_span * c.Clda / jx
        if abs(a_phi2) < 1e-9:
            raise ValueError(
                "Airframe has no roll control authority (Clda is zero) — "
                "the autopilot cannot stabilise it."
            )
        self.kp_roll = float(np.sign(a_phi2) * g.max_surface / g.max_roll_error)
        omega_phi = float(np.sqrt(abs(a_phi2 * self.kp_roll)))
        self.kd_roll = self._damping_gain(2.0 * g.roll_damping * omega_phi, a_phi1, a_phi2)

        # ── pitch loop: theta'' = -a_theta1 theta' - a_theta2 theta ────────
        #                          + a_theta3 delta_e
        a_theta1 = -qbar_s * p.chord * c.Cmq * p.chord / (2.0 * va) / jy
        a_theta2 = -qbar_s * p.chord * c.Cma / jy
        a_theta3 = qbar_s * p.chord * c.Cmde / jy
        if abs(a_theta3) < 1e-9:
            raise ValueError(
                "Airframe has no pitch control authority (Cmde is zero) — "
                "the autopilot cannot stabilise it."
            )
        self.kp_pitch = float(np.sign(a_theta3) * g.max_surface / g.max_pitch_error)
        omega_theta = float(np.sqrt(max(a_theta2 + self.kp_pitch * a_theta3, 1e-9)))
        self.kd_pitch = self._damping_gain(2.0 * g.pitch_damping * omega_theta, a_theta1, a_theta3)

        self.roll_bandwidth = omega_phi
        self.pitch_bandwidth = omega_theta

        # DC gain from commanded pitch to achieved pitch, used to scale the
        # altitude loop so it does not fight the inner loop's droop.
        self._pitch_dc = (self.kp_pitch * a_theta3) / (a_theta2 + self.kp_pitch * a_theta3)

        # ── outer loops ───────────────────────────────────────────────────
        # Course: banking produces turn rate g*tan(phi)/Va, so the plant
        # from roll command to course is an integrator with gain g/Va.
        self.kp_course = g.max_roll / g.max_course_error
        omega_chi = (p.gravity / va) * self.kp_course / (2.0 * g.course_damping)
        self.ki_course = omega_chi**2 * va / p.gravity

        # Altitude: pitch produces climb rate Va*theta, another integrator,
        # de-rated by the inner pitch loop's DC droop.
        climb_gain = self._pitch_dc * va
        self.kp_altitude = g.max_pitch / g.max_altitude_error
        omega_h = climb_gain * self.kp_altitude / (2.0 * g.altitude_damping)
        self.ki_altitude = omega_h**2 / climb_gain

        self.course_bandwidth = omega_chi
        self.altitude_bandwidth = omega_h

    def reset(self) -> None:
        """Zero every integrator."""
        self._int_course = 0.0
        self._int_altitude = 0.0
        self._int_airspeed = 0.0
        self._yaw_washout = 0.0
        self.diagnostics = AutopilotDiagnostics()

    # ── control ───────────────────────────────────────────────────────────

    @staticmethod
    def _pi_step(
        accumulator: float,
        error: float,
        dt: float,
        kp: float,
        ki: float,
        limit: float,
    ) -> tuple[float, float]:
        """One PI update with conditional integration.

        Returns ``(output, new_accumulator)``. The integrator only advances
        while the *unsaturated* output is inside ``[-limit, limit]``, and the
        stored value is additionally clamped so its own contribution can
        never exceed the output range. Clamping the raw integral instead —
        the usual shortcut — still lets a large ``ki`` push a saturated
        command far past the limit and then take seconds to unwind.
        """
        candidate = accumulator + error * dt
        output = kp * error + ki * candidate
        if abs(output) > limit:
            # Saturated: hold the integrator unless the error is now
            # driving the output back into range.
            if np.sign(error) == np.sign(output):
                candidate = accumulator
                output = kp * error + ki * candidate
        if ki > 1e-12:
            candidate = float(np.clip(candidate, -limit / ki, limit / ki))
        return float(np.clip(output, -limit, limit)), float(candidate)

    def compute(
        self,
        state: NDArray[np.floating],
        command: AutopilotCommand,
        dt: float,
    ) -> NDArray[np.floating]:
        """Return ``[elevator, aileron, rudder, throttle]`` for one step.

        Parameters
        ----------
        state
            Aircraft state vector, 12 elements, FLU convention.
        command
            Altitude / airspeed / course setpoints.
        dt
            Control period [s].
        """
        g = self.gains
        phi = float(state[3])
        # Convert to the aerospace convention the loop design assumes.
        theta_up = -float(state[4])
        roll_rate = float(state[9])
        pitch_rate_up = -float(state[10])
        yaw_rate_aero = -float(state[11])
        altitude = float(state[2])
        airspeed = float(np.linalg.norm(state[6:9]))

        # ── outer loop: course → roll command ─────────────────────────────
        # Feed back *course* — the direction the aircraft is actually
        # travelling — not heading. The two differ by the sideslip angle,
        # and the plant model behind this loop (turn rate = g tan(phi)/Va)
        # describes the velocity vector, not where the nose points. Closing
        # the loop on heading instead leaves the nose hunting around a
        # perfectly good flight path, which on a rudderless airframe never
        # settles.
        velocity_world = euler_to_rotation(*state[3:6]) @ state[6:9]
        course = float(np.arctan2(velocity_world[1], velocity_world[0]))
        course_error = float(
            np.arctan2(np.sin(command.course - course), np.cos(command.course - course))
        )
        # A guidance layer that already knows the aircraft is about to fly a
        # constant-radius turn can say so, and the PI loop is then left with
        # only the error around it. The PI gets whatever bank authority the
        # feed-forward has not already spent, so its own anti-windup still
        # measures the real remaining range rather than the full envelope.
        feedforward = float(np.clip(command.roll_feedforward, -g.max_roll, g.max_roll))
        authority = max(g.max_roll - abs(feedforward), _MIN_PI_AUTHORITY)
        roll_cmd, self._int_course = self._pi_step(
            self._int_course, course_error, dt, self.kp_course, self.ki_course, authority
        )
        # ENU yaw increases counter-clockwise while banking right turns the
        # aircraft clockwise, hence the sign flip. The final clamp is what
        # actually guarantees the envelope: the authority split above keeps
        # the integrator honest, but its floor means the two terms can still
        # sum past the limit when the feed-forward alone is already at it.
        roll_cmd = float(np.clip(-roll_cmd + feedforward, -g.max_roll, g.max_roll))

        # ── outer loop: altitude → pitch command ──────────────────────────
        altitude_error = command.altitude - altitude
        pitch_cmd, self._int_altitude = self._pi_step(
            self._int_altitude,
            altitude_error,
            dt,
            self.kp_altitude,
            self.ki_altitude,
            g.max_pitch,
        )

        # Bank costs vertical lift; add the extra pitch a level turn needs.
        pitch_cmd += (1.0 / max(np.cos(phi), 0.3) - 1.0) * np.radians(4.0)

        # ── inner loops ───────────────────────────────────────────────────
        aileron = self.kp_roll * (roll_cmd - phi) - self.kd_roll * roll_rate
        elevator = self.kp_pitch * (pitch_cmd - theta_up) - self.kd_pitch * pitch_rate_up

        # ── airspeed → throttle ───────────────────────────────────────────
        airspeed_error = command.airspeed - airspeed
        throttle_correction, self._int_airspeed = self._pi_step(
            self._int_airspeed,
            airspeed_error,
            dt,
            g.airspeed_kp,
            g.airspeed_ki,
            1.0,
        )
        throttle = self._trim_throttle + throttle_correction

        # ── sideslip + yaw damper → rudder ────────────────────────────────
        # Positive rudder yaws whichever way Cndr says it does; following
        # its sign makes both loops work on any airframe, and disables the
        # rudder entirely on a rudderless flying wing (Cndr == 0).
        beta = flow_state(flu_to_frd(state[6:9]), self.params.rho_air).beta
        rudder_sign = float(np.sign(self.params.coeffs.Cndr))
        # Washout so the damper fights Dutch roll but not a steady turn:
        # a constant yaw rate decays out of the filter state within a few
        # time constants, leaving the turn uncommanded.
        alpha_washout = dt / max(g.yaw_washout_tau + dt, 1e-9)
        self._yaw_washout += alpha_washout * (yaw_rate_aero - self._yaw_washout)
        yaw_transient = yaw_rate_aero - self._yaw_washout
        rudder = rudder_sign * (g.sideslip_kp * beta - g.yaw_damper_kd * yaw_transient)

        raw = np.array([elevator, aileron, rudder, throttle])
        clipped = np.array(
            [
                np.clip(elevator, -g.max_surface, g.max_surface),
                np.clip(aileron, -g.max_surface, g.max_surface),
                np.clip(rudder, -g.max_surface, g.max_surface),
                np.clip(throttle, 0.0, 1.0),
            ]
        )

        self.diagnostics = AutopilotDiagnostics(
            roll_cmd=roll_cmd,
            pitch_cmd=pitch_cmd,
            altitude_error=altitude_error,
            airspeed_error=airspeed_error,
            course_error=course_error,
            saturated=bool(np.any(np.abs(raw - clipped) > 1e-9)),
        )
        return clipped
