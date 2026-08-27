# Erwin Lejeune - 2026-08-18
"""Straight-line and orbit vector fields for fixed-wing path following.

Reference: R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
Practice*, Princeton University Press, 2012, Chapter 10 (eqs. 10.8 for the
straight line, 10.13 for the orbit).

Both laws are **vector fields**: given where the aircraft is relative to a
geometric path, they return the course it should be holding. They do not
point the nose at a waypoint. Chasing a point leaves a standing lateral
offset whenever anything pushes the aircraft sideways, because the law is
satisfied the moment the nose is on the target no matter how far off the
path the aircraft has drifted. A cross-track law is only satisfied when the
aircraft is *on the line*.

Sign conventions
----------------
The textbook is written in **NED**, where course increases clockwise. This
library is **ENU**, where course increases counter-clockwise
(``atan2(v_y, v_x)``, measured from +x/east). Every course formula
therefore needs converting at the boundary — see
:doc:`/guide/conventions`.

The pleasant surprise is that both laws come out *algebraically identical*
after the conversion, because each involves two sign flips that cancel:

``straight line``
    The book's lateral path-frame error ``e_py`` is positive when the
    aircraft is to the **right** of the path (NED path frame). Flipping to
    ENU flips the sense of course, and measuring the error positive to the
    **left** flips it back. So :meth:`LinePath.command` uses the book's
    expression verbatim with a left-positive cross-track error.

``orbit``
    The book's ``lambda`` is ``+1`` for a **clockwise** orbit viewed from
    above. In ENU the same expression works with ``lambda = +1`` meaning
    **counter-clockwise**, which is what :class:`OrbitDirection` encodes.

Both are stated as claims in the module docstring rather than left implicit
because they are exactly the kind of thing that produces a controller which
flies a stable, confident circle in the wrong direction.

Gains are derived, not tuned
----------------------------
Following the same philosophy as
:class:`~flybots.control.fixed_wing_autopilot.FixedWingAutopilot`, the
convergence gains are computed rather than hand-picked, and by the same
argument the autopilot uses on its own cascade: **loop separation**. This
guidance sits one loop outside the autopilot's course loop, so it has to be
slower than it.

Linearising the straight-line law near the path gives

.. math::
    \\dot e = V_g \\sin(\\chi - \\chi_q)
           \\approx -V_g \\chi_\\infty \\frac{2}{\\pi} k_{path}\\, e

a first-order response with pole :math:`\\omega_e = 2 V_g \\chi_\\infty
k_{path} / \\pi`. Setting that to the autopilot's course-loop bandwidth
divided by :attr:`GuidanceGains.loop_separation` and solving for
:math:`k_{path}` is the whole derivation — see
:meth:`GuidanceGains.line_gain`. The orbit gain falls out of the identical
linearisation about the circle, so one design number sets both.

The course-loop bandwidth itself is read back out of
:class:`~flybots.control.fixed_wing_autopilot.AutopilotGains`, evaluated at
the *commanded* airspeed of the leg being flown rather than at cruise. The
autopilot fixes its own gains at cruise, so an aircraft flying a slow leg
genuinely has a faster course loop than its design point — and the layer
above should know that.

The other quantity that matters is the **bank-limited turn radius**

.. math:: R_{min} = \\frac{V_a^2}{g \\tan\\phi_{max}}

which sets the minimum flyable orbit (:meth:`OrbitPath.validate`) and the
fillet distances the mission layer switches waypoints on. It also gives a
sanity floor for the derivation above: an aircraft crossing a line at
:math:`\\chi_\\infty` sweeps :math:`R_{min}(1 - \\cos\\chi_\\infty)` of
lateral distance merely rolling out onto it, so a transition distance below
that is asking for something the roll limit forbids. See
:meth:`GuidanceGains.rollout_width`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flybots.control.fixed_wing_autopilot import AutopilotGains

__all__ = [
    "GuidanceError",
    "GuidanceGains",
    "GuidanceOutput",
    "LinePath",
    "OrbitDirection",
    "OrbitPath",
    "minimum_turn_radius",
]

# Below this, a direction vector is numerically indistinguishable from zero
# and normalising it would amplify round-off into a random heading.
_EPS = 1e-9


class GuidanceError(ValueError):
    """Raised when a requested path cannot be flown by the airframe.

    The commonest case is an orbit radius tighter than the bank-limited
    turn radius at the commanded airspeed. Returning a plausible-looking
    command for a path the aircraft physically cannot hold is worse than
    refusing: the aircraft flies a larger circle than asked and the
    guidance quietly saturates, which looks like a tuning problem.
    """


class OrbitDirection(IntEnum):
    """Direction of travel around an orbit, viewed from above in ENU.

    The integer value is the ``lambda`` of Beard & McLain eq. 10.13, with
    the sense flipped for ENU (the book's ``+1`` is clockwise in NED).
    """

    COUNTER_CLOCKWISE = 1
    CLOCKWISE = -1


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _unit(vector: NDArray[np.floating]) -> NDArray[np.floating]:
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        raise GuidanceError("direction vector is zero — cannot define a path from it")
    return np.asarray(vector, dtype=float) / norm


def minimum_turn_radius(
    airspeed: float,
    bank_limit: float,
    gravity: float = 9.81,
) -> float:
    """Radius [m] of the tightest level turn the airframe can hold.

    ``R = Va^2 / (g tan(phi_max))`` — the coordinated-turn relation, which
    is also what sets the turn radius in Beard & McLain's Dubins path
    construction (Chapter 11).

    Parameters
    ----------
    airspeed
        True airspeed [m/s]. Without wind this is also the ground speed,
        which is what the turn geometry is actually about.
    bank_limit
        Maximum roll angle [rad] the autopilot will command, strictly
        inside ``(0, pi/2)``.
    gravity
        Gravitational acceleration [m/s^2].
    """
    if not 0.0 < bank_limit < np.pi / 2:
        raise GuidanceError(f"bank limit must be in (0, pi/2) rad, got {bank_limit}")
    if airspeed <= 0.0:
        raise GuidanceError(f"airspeed must be positive, got {airspeed}")
    return float(airspeed**2 / (gravity * np.tan(bank_limit)))


@dataclass
class GuidanceGains:
    """Design quantities for the two vector fields.

    As with :class:`~flybots.control.fixed_wing_autopilot.AutopilotGains`,
    these are the *design* parameters — an intercept angle and a loop
    separation — and the convergence gains are derived from them, the
    airframe, and the autopilot underneath.
    """

    course_infinity: float = np.radians(60.0)
    """Maximum intercept angle far from a straight line [rad].

    ``chi_inf`` in the book. Must be in ``(0, pi/2)``: at exactly ``pi/2``
    the aircraft is commanded to fly perpendicular to the path, which it
    can never roll out of in finite lateral distance.
    """

    loop_separation: float = 2.0
    """How many times slower than the course loop this guidance runs.

    The one number that sets both convergence gains. Successive loop
    closure needs the outer loop slower than the inner one, and this layer
    is the loop outside the autopilot's course hold. Below ``1`` the
    guidance asks for course changes faster than the autopilot can deliver
    and the aircraft weaves across the path; far above it, capture is
    needlessly leisurely.

    ``2.0`` is picked with the measured numbers in hand rather than from
    the theory alone. Capturing a line from five turn radii off, overshoot
    is already zero at ``1.5`` on all four presets and only appears below
    it — the mini trainer overshoots 3.4 m, 23 % of its own turn radius, at
    ``1.0``. ``2.0`` keeps a third again of margin on that boundary and
    costs about 1.5x the capture time. The full table is on the docs page.
    """

    autopilot: AutopilotGains = field(default_factory=AutopilotGains)
    """The autopilot's own design gains, read but never modified.

    Two things are taken from here: the roll limit that fixes the turn
    radius, and the course-loop bandwidth this layer has to stay below.
    Holding a reference rather than copying the numbers is what stops the
    two layers drifting apart — retune the autopilot and the guidance
    retunes with it. Pass the *same* instance you give
    :class:`~flybots.control.fixed_wing_autopilot.FixedWingAutopilot`.
    """

    gravity: float = 9.81
    """Gravitational acceleration [m/s^2]."""

    def __post_init__(self) -> None:
        if not 0.0 < self.course_infinity < np.pi / 2:
            raise GuidanceError(
                f"course_infinity must be in (0, pi/2) rad, got {self.course_infinity}"
            )
        if self.loop_separation <= 0.0:
            raise GuidanceError(f"loop_separation must be positive, got {self.loop_separation}")

    # ── derived quantities ────────────────────────────────────────────────

    def turn_radius(self, airspeed: float) -> float:
        """Bank-limited turn radius [m] at ``airspeed``."""
        return minimum_turn_radius(airspeed, self.autopilot.max_roll, self.gravity)

    def course_bandwidth(self, airspeed: float) -> float:
        """Bandwidth [rad/s] of the autopilot's course loop at ``airspeed``.

        Reproduces the autopilot's own design relation — banking gives a
        turn rate ``g tan(phi) / Va``, so the plant from roll command to
        course is an integrator of gain ``g / Va``, and closing a PI loop
        of proportional gain ``kp`` around it at damping ``zeta`` puts the
        natural frequency at ``(g / Va) kp / (2 zeta)``.

        Evaluated at the airspeed actually being flown, not at cruise. The
        autopilot freezes ``kp_course`` at its design point, so the loop it
        really has is faster on a slow leg and slower on a fast one, and
        the layer above should size itself against the real one.
        """
        gains = self.autopilot
        kp_course = gains.max_roll / gains.max_course_error
        return float((self.gravity / airspeed) * kp_course / (2.0 * gains.course_damping))

    def convergence_bandwidth(self, airspeed: float) -> float:
        """Target bandwidth [rad/s] for closing on a path at ``airspeed``."""
        return self.course_bandwidth(airspeed) / self.loop_separation

    def rollout_width(self, airspeed: float) -> float:
        """Lateral distance [m] swept while rolling out from a full intercept.

        An aircraft crossing at ``course_infinity`` to the path and turning
        at its bank limit traces an arc of radius ``R``; the lateral
        distance between the start of that turn and being parallel to the
        path is ``R (1 - cos(chi_inf))``. No guidance law can capture a
        line from closer than this without overshooting, whatever its
        gains, so it is the floor that
        ``1 / line_gain`` has to clear — there is a test asserting it does,
        for every preset.
        """
        return float(self.turn_radius(airspeed) * (1.0 - np.cos(self.course_infinity)))

    def line_gain(self, airspeed: float) -> float:
        """``k_path`` [1/m] for the straight-line law at ``airspeed``.

        Inverts the linearised cross-track pole
        ``omega_e = 2 Vg chi_inf k_path / pi`` for ``k_path``. The
        reciprocal is the cross-track error at which the law commands half
        of ``course_infinity``, which is the number worth looking at when
        judging whether a capture will be tight or lazy.
        """
        omega = self.convergence_bandwidth(airspeed)
        return float(np.pi * omega / (2.0 * airspeed * self.course_infinity))

    def orbit_gain(self, radius: float, airspeed: float) -> float:
        """``k_orbit`` [-] for the orbit law at ``radius`` and ``airspeed``.

        The same linearisation taken about the circle instead of the line:
        with radial error ``eps``, ``d(eps)/dt = -Vg sin(arctan(k eps /
        rho)) ~= -Vg k eps / rho``, a first-order response with pole
        ``Vg k / rho``. Setting that to the same convergence bandwidth
        gives ``k = rho omega_e / Vg``, so one design number sets both
        laws and a straight leg and an orbit converge at the same rate.
        """
        return float(radius * self.convergence_bandwidth(airspeed) / airspeed)

    def transition_distance(self, airspeed: float) -> float:
        """Cross-track error [m] at which half of ``course_infinity`` is commanded."""
        return 1.0 / max(self.line_gain(airspeed), _EPS)


@dataclass(frozen=True)
class GuidanceOutput:
    """What a vector field asks the autopilot for at one instant."""

    course: float
    """Commanded course [rad], ENU (counter-clockwise from +x)."""
    altitude: float
    """Commanded altitude [m]."""
    airspeed: float
    """Commanded true airspeed [m/s]."""
    roll_feedforward: float = 0.0
    """Coordinated-turn bank [rad] this path is known to require.

    Zero on a straight line, which is not curved. On an orbit it is the
    bank a level turn of that radius needs, and handing it to the autopilot
    is what stops every turn entry bulging outward while the course loop's
    integrator charges — see
    :attr:`~flybots.control.fixed_wing_autopilot.AutopilotCommand.roll_feedforward`.
    """
    path_error: float = 0.0
    """Signed distance from the path [m].

    For a :class:`LinePath` this is the cross-track error, positive when
    the aircraft is to the **left** of the path direction. For an
    :class:`OrbitPath` it is the radial error ``d - radius``, positive
    outside the circle.
    """


@dataclass
class LinePath:
    """An infinite straight line in 3-D, followed by a cross-track law.

    Parameters
    ----------
    origin
        Any point on the line [m], world ENU.
    direction
        Direction of travel along the line; normalised on construction and
        need not be a unit vector. Its ``z`` component is the climb
        gradient, so a line between waypoints at different altitudes
        commands a steady climb.
    airspeed
        Commanded true airspeed [m/s] along this line.

    Examples
    --------
    >>> import numpy as np
    >>> from flybots.guidance.fixed_wing_paths import GuidanceGains, LinePath
    >>> line = LinePath.between([0.0, 0.0, 100.0], [1000.0, 0.0, 100.0], airspeed=35.0)
    >>> float(np.degrees(line.course))
    0.0
    >>> # 50 m to the left of the line: steer right, so course goes negative.
    >>> out = line.command(np.array([100.0, 50.0, 100.0]), GuidanceGains())
    >>> bool(out.course < 0.0)
    True
    """

    origin: NDArray[np.floating]
    direction: NDArray[np.floating]
    airspeed: float

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float).reshape(3).copy()
        self.direction = _unit(np.asarray(self.direction, dtype=float).reshape(3))
        if self.airspeed <= 0.0:
            raise GuidanceError(f"airspeed must be positive, got {self.airspeed}")
        if float(np.hypot(self.direction[0], self.direction[1])) < _EPS:
            raise GuidanceError(
                "a straight-line path needs a horizontal component — a fixed wing "
                "cannot fly a vertical line"
            )

    @classmethod
    def between(cls, start: ArrayLike, end: ArrayLike, airspeed: float) -> LinePath:
        """Line from ``start`` through ``end`` (both 3-vectors, ENU)."""
        start_arr = np.asarray(start, dtype=float).reshape(3)
        end_arr = np.asarray(end, dtype=float).reshape(3)
        return cls(origin=start_arr, direction=end_arr - start_arr, airspeed=airspeed)

    # ── geometry ──────────────────────────────────────────────────────────

    @property
    def course(self) -> float:
        """Course of the line [rad], ENU."""
        return float(np.arctan2(self.direction[1], self.direction[0]))

    @property
    def climb_gradient(self) -> float:
        """Altitude gained per metre of horizontal travel along the line."""
        horizontal = float(np.hypot(self.direction[0], self.direction[1]))
        return float(self.direction[2] / max(horizontal, _EPS))

    def _horizontal_unit(self) -> NDArray[np.floating]:
        return _unit(np.array([self.direction[0], self.direction[1]]))

    def cross_track(self, position: ArrayLike) -> float:
        """Signed lateral distance [m] from the line, positive to its **left**.

        Measured in the horizontal plane, which is what the course loop can
        act on; the vertical component is handled separately by
        :meth:`altitude_at`.
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        along = self._horizontal_unit()
        # Rotating the path direction +90 deg in ENU points to its left.
        left = np.array([-along[1], along[0]])
        return float(left @ (pos[:2] - self.origin[:2]))

    def along_track(self, position: ArrayLike) -> float:
        """Signed distance [m] travelled along the line from ``origin``."""
        pos = np.asarray(position, dtype=float).reshape(3)
        return float(self._horizontal_unit() @ (pos[:2] - self.origin[:2]))

    def altitude_at(self, position: ArrayLike) -> float:
        """Altitude [m] of the line abeam ``position``.

        Beard & McLain eq. 10.5 projects the aircraft onto the path and
        commands the altitude of the projection, which is what makes a
        climbing leg ramp rather than step.
        """
        return float(self.origin[2] + self.along_track(position) * self.climb_gradient)

    # ── the vector field ──────────────────────────────────────────────────

    def command(self, position: ArrayLike, gains: GuidanceGains) -> GuidanceOutput:
        """Course, altitude and airspeed to hold at ``position``.

        Implements Beard & McLain eq. 10.8,

        .. math::
            \\chi^c = \\chi_q - \\chi_\\infty \\frac{2}{\\pi}
                      \\arctan(k_{path}\\, e)

        with ``e`` the left-positive cross-track error of
        :meth:`cross_track`. Far from the line the command saturates at
        ``chi_inf`` off the path course; on the line it *is* the path
        course. The bounded intercept angle is what keeps the law stable
        even where a linear cross-track gain would demand a course the
        aircraft cannot roll out of.
        """
        error = self.cross_track(position)
        gain = gains.line_gain(self.airspeed)
        intercept = gains.course_infinity * (2.0 / np.pi) * np.arctan(gain * error)
        return GuidanceOutput(
            course=_wrap_pi(self.course - intercept),
            altitude=self.altitude_at(position),
            airspeed=self.airspeed,
            path_error=error,
        )


@dataclass
class OrbitPath:
    """A level circular orbit, followed by the orbit vector field.

    Parameters
    ----------
    centre
        Orbit centre [m], world ENU. Its ``z`` is the orbit altitude.
    radius
        Commanded radius [m]. Must be at least the bank-limited turn radius
        at ``airspeed`` or the aircraft cannot hold it — see
        :meth:`validate`.
    direction
        Which way round, viewed from above.
    airspeed
        Commanded true airspeed [m/s].

    Examples
    --------
    >>> import numpy as np
    >>> from flybots.guidance.fixed_wing_paths import (
    ...     GuidanceGains, OrbitDirection, OrbitPath)
    >>> orbit = OrbitPath([0.0, 0.0, 100.0], radius=200.0,
    ...                   direction=OrbitDirection.COUNTER_CLOCKWISE, airspeed=35.0)
    >>> # Sitting exactly on the circle due east of the centre, a
    >>> # counter-clockwise orbit is flown heading north (+90 deg).
    >>> out = orbit.command(np.array([200.0, 0.0, 100.0]), GuidanceGains())
    >>> round(float(np.degrees(out.course)), 6)
    90.0
    """

    centre: NDArray[np.floating]
    radius: float
    direction: OrbitDirection
    airspeed: float

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=float).reshape(3).copy()
        self.radius = float(self.radius)
        self.direction = OrbitDirection(self.direction)
        if self.radius <= 0.0:
            raise GuidanceError(f"orbit radius must be positive, got {self.radius}")
        if self.airspeed <= 0.0:
            raise GuidanceError(f"airspeed must be positive, got {self.airspeed}")

    # ── geometry ──────────────────────────────────────────────────────────

    @property
    def altitude(self) -> float:
        """Orbit altitude [m]."""
        return float(self.centre[2])

    def validate(self, gains: GuidanceGains) -> None:
        """Raise :class:`GuidanceError` if the airframe cannot hold this orbit.

        A level turn of radius ``R`` at airspeed ``Va`` needs a bank angle
        ``atan(Va^2 / (g R))``. If that exceeds the autopilot's roll limit
        the aircraft simply flies a wider circle, so the commanded radius
        is a fiction. Catching it here turns a silent tracking error into
        an error message naming the smallest radius that *is* flyable.
        """
        minimum = gains.turn_radius(self.airspeed)
        if self.radius < minimum:
            required = np.degrees(np.arctan(self.airspeed**2 / (gains.gravity * self.radius)))
            raise GuidanceError(
                f"orbit radius {self.radius:.1f} m is below the bank-limited turn "
                f"radius {minimum:.1f} m at {self.airspeed:.1f} m/s — holding it would "
                f"need {required:.1f} deg of bank against a "
                f"{np.degrees(gains.autopilot.max_roll):.1f} deg limit"
            )

    def angle_at(self, position: ArrayLike) -> float:
        """Polar angle [rad] of ``position`` about the centre, ENU.

        Measured counter-clockwise from +x, so it increases along a
        counter-clockwise orbit.
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        offset = pos[:2] - self.centre[:2]
        return float(np.arctan2(offset[1], offset[0]))

    def radial_distance(self, position: ArrayLike) -> float:
        """Horizontal distance [m] from the orbit centre."""
        pos = np.asarray(position, dtype=float).reshape(3)
        return float(np.linalg.norm(pos[:2] - self.centre[:2]))

    def radial_error(self, position: ArrayLike) -> float:
        """``d - radius`` [m]; positive outside the circle."""
        return self.radial_distance(position) - self.radius

    # ── the vector field ──────────────────────────────────────────────────

    def command(self, position: ArrayLike, gains: GuidanceGains) -> GuidanceOutput:
        """Course, altitude and airspeed to hold at ``position``.

        Implements Beard & McLain eq. 10.13,

        .. math::
            \\chi^c = \\theta + \\lambda\\left[\\frac{\\pi}{2}
                      + \\arctan\\!\\left(k_{orbit}\\frac{d - \\rho}{\\rho}\\right)\\right]

        with ``theta`` the polar angle of the aircraft about the centre and
        ``lambda = +1`` for counter-clockwise in ENU. On the circle the
        command is the tangent; far outside it saturates at straight
        towards the centre, and at the centre it points straight out.

        The output also carries the **coordinated-turn bank** this circle
        needs, ``-lambda atan(Va^2 / (g rho))``, negative for a left
        (counter-clockwise) turn in this library's ENU frame. It is not
        there to hold the radius — a settled orbit needs no help, and holds
        to under a millimetre in the tests. It is there for the *entry*: a
        PI course loop driving an integrator plant reaches a constant turn
        rate only once its integrator has charged, and on a circuit of
        alternating straights and turns the aircraft never gets that long.
        Beard & McLain add the identical term to their orbit follower.
        """
        angle = self.angle_at(position)
        error = self.radial_error(position)
        gain = gains.orbit_gain(self.radius, self.airspeed)
        offset = np.pi / 2.0 + np.arctan(gain * error / self.radius)
        bank = np.arctan(self.airspeed**2 / (gains.gravity * self.radius))
        return GuidanceOutput(
            course=_wrap_pi(angle + int(self.direction) * offset),
            altitude=self.altitude,
            airspeed=self.airspeed,
            roll_feedforward=float(-int(self.direction) * bank),
            path_error=error,
        )
