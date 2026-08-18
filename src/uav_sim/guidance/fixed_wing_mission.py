# Erwin Lejeune - 2026-08-18
"""Mission sequencing for fixed-wing aircraft: waypoints, patterns, RTL.

Reference: R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
Practice*, Princeton University Press, 2012, Chapter 11 (waypoint following,
half-plane switching in eq. 11.1, and the fillet construction of section
11.2). The straight-line and orbit vector fields this module sequences live
in :mod:`~uav_sim.guidance.fixed_wing_paths`.

Where this deviates from the book
---------------------------------
* **Half-plane *or* capture radius.** The book switches waypoints purely on
  the half-plane test. That is what makes overshoot recoverable, but on its
  own it means the aircraft flies to the waypoint and only then starts
  turning, bulging outside every corner. Adding a capture radius derived
  from the fillet geometry of section 11.2 lets it start the turn early,
  while the half-plane stays as the guarantee. The two are combined with
  **or**, which is the point: a radius alone lets an aircraft that
  overshoots orbit the waypoint forever hunting for a circle it can no
  longer enter, and a half-plane alone corners badly.
* **Fillets are used as an acceptance radius, not as an arc.** The book
  inserts an explicit circular fillet path between two straight legs. Here
  the fillet radius only decides *when* to switch legs, and the turn itself
  is flown by the straight-line vector field of the next leg pulling the
  aircraft in. Fewer path types, one less place for the geometry to be
  wrong, and the resulting corner is within a few metres of the fillet arc
  because the acceptance point is exactly where the arc would have started.
* **Return-to-launch is not in the book at all.** It is assembled here from
  a straight leg and a loiter orbit; see :func:`return_to_launch_plan`.
* **Turn bank is fed forward.** Every command carries the coordinated-turn
  bank the active path requires — zero on a straight, the level-turn bank
  on an orbit. Without it a circuit of alternating straights and turns
  never leaves the transient: the course PI has to re-charge its integrator
  at every hand-over, and the aircraft bulges out of each turn entry and
  overshoots each exit. On a mini-trainer racetrack of 2.5 turn radii that
  is the difference between 18.3 m and 6.3 m of worst-case path error.

Wind
----
The airframe model in :mod:`uav_sim.vehicles.fixed_wing` has **no wind
field** — its body velocity is both the air-relative and the inertial
velocity, and adding wind is a separate open roadmap item. So nothing here
is tested against wind, and no claim is made that it is wind-compensating.

What *is* true is that the layer is built so wind enters in one place when
it lands. The guidance closes on inertial cross-track distance and commands
**course**, not heading, and the autopilot behind it also tracks course. A
crabbing aircraft therefore already has the right feedback structure: the
nose points wherever it must for the *velocity vector* to lie along the
path. The waypoint acceptance test is likewise the reason the phrase "so it
cannot circle a waypoint it overshot in wind" appears in the roadmap — the
half-plane is what makes the overshoot recoverable, whatever caused it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uav_sim.control.fixed_wing_autopilot import AutopilotCommand
from uav_sim.guidance.fixed_wing_paths import (
    GuidanceError,
    GuidanceGains,
    GuidanceOutput,
    LinePath,
    OrbitDirection,
    OrbitPath,
)
from uav_sim.vehicles.fixed_wing.fixed_wing import FixedWingParams

__all__ = [
    "FixedWingMission",
    "LineLeg",
    "MissionDiagnostics",
    "MissionLeg",
    "OrbitLeg",
    "orbit_plan",
    "racetrack_plan",
    "return_to_launch_plan",
    "waypoint_plan",
    "waypoint_reached",
]

_EPS = 1e-9

# A fillet radius is `R tan(delta/2)`, which runs away as the turn angle
# approaches a full reversal. Past about 143 deg of turn there is no arc
# that joins the two legs inside a sane distance, so the acceptance radius
# is capped at this multiple of the turn radius and the half-plane test
# takes over as the thing that actually ends the leg.
_MAX_FILLET_RADII = 3.0


def _horizontal_unit(vector: ArrayLike) -> NDArray[np.floating]:
    """Unit horizontal direction of a 2- or 3-vector."""
    arr = np.asarray(vector, dtype=float).reshape(-1)[:2]
    norm = float(np.linalg.norm(arr))
    if norm < _EPS:
        raise GuidanceError("cannot take the horizontal direction of a vertical segment")
    return arr / norm


def _turn_angle(incoming: ArrayLike, outgoing: ArrayLike) -> float:
    """Angle [rad] in ``[0, pi]`` between two horizontal directions."""
    a, b = _horizontal_unit(incoming), _horizontal_unit(outgoing)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def _bisector_normal(incoming: ArrayLike, outgoing: ArrayLike) -> NDArray[np.floating]:
    """Unit normal of the switching half-plane between two legs.

    Beard & McLain eq. 11.1: the plane through the waypoint whose normal
    bisects the incoming and outgoing path directions. A near-reversal
    makes the bisector vanish, in which case the incoming direction is the
    only sensible normal — the aircraft has arrived once it is past the
    waypoint, and there is no corner to bisect.
    """
    a, b = _horizontal_unit(incoming), _horizontal_unit(outgoing)
    total = a + b
    norm = float(np.linalg.norm(total))
    return a if norm < 1e-6 else total / norm


def waypoint_reached(
    position: ArrayLike,
    waypoint: ArrayLike,
    half_plane_normal: ArrayLike,
    capture_radius: float,
) -> bool:
    """Has the aircraft finished with this waypoint?

    True when **either** test fires:

    ``capture radius``
        The aircraft is horizontally within ``capture_radius`` of the
        waypoint. This is what lets it cut the corner instead of flying to
        the point and pivoting.
    ``half-plane``
        The aircraft has crossed the plane through the waypoint whose
        normal is ``half_plane_normal`` — Beard & McLain eq. 11.1.

    The half-plane is the one that cannot be defeated. An aircraft blown
    wide of a waypoint may never enter the capture radius at all; with a
    radius test alone it turns back, misses again, and orbits the waypoint
    indefinitely. Once it is past the plane it is past the waypoint, and
    the mission moves on.

    Both tests are **horizontal**. A leg is finished when the aircraft is
    past it in plan view; altitude is tracked continuously by the line's
    own altitude command and is not a gate, because a fixed wing that has
    not finished climbing must not stop navigating.
    """
    offset = (
        np.asarray(position, dtype=float).reshape(-1)[:2]
        - np.asarray(waypoint, dtype=float).reshape(-1)[:2]
    )
    if float(np.linalg.norm(offset)) <= capture_radius:
        return True
    normal = np.asarray(half_plane_normal, dtype=float).reshape(-1)[:2]
    return bool(normal @ offset >= 0.0)


class MissionLeg(ABC):
    """One stage of a mission: a geometric path plus a termination test.

    The split is deliberate. Everything about *where the path is* lives in
    :mod:`~uav_sim.guidance.fixed_wing_paths` and is stateless and pure;
    everything about *which leg we are on* lives in
    :class:`FixedWingMission`. A leg holds neither — it is the immutable
    description of a stage, so the same leg object can be replayed,
    inspected or plotted without a simulation having touched it.
    """

    label: str

    @property
    @abstractmethod
    def path(self) -> LinePath | OrbitPath:
        """The geometric path this leg follows."""

    @abstractmethod
    def command(self, position: ArrayLike, gains: GuidanceGains) -> GuidanceOutput:
        """Guidance command at ``position``."""

    @abstractmethod
    def is_complete(self, position: ArrayLike, angle_flown: float) -> bool:
        """Whether the aircraft is finished with this leg.

        ``angle_flown`` is the signed angle [rad] swept about an orbit
        centre since the leg was entered, positive in the commanded
        direction. It is ignored by straight legs.
        """


@dataclass
class LineLeg(MissionLeg):
    """A straight leg ending at a waypoint."""

    line: LinePath
    target: NDArray[np.floating]
    """Waypoint [m] the leg ends at, world ENU."""
    half_plane_normal: NDArray[np.floating]
    """Unit horizontal normal of the switching plane through ``target``."""
    capture_radius: float
    """Horizontal acceptance radius [m] around ``target``."""
    label: str = "line"

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=float).reshape(3).copy()
        self.half_plane_normal = _horizontal_unit(self.half_plane_normal)
        self.capture_radius = max(0.0, float(self.capture_radius))

    @property
    def path(self) -> LinePath:
        return self.line

    def command(self, position: ArrayLike, gains: GuidanceGains) -> GuidanceOutput:
        return self.line.command(position, gains)

    def is_complete(self, position: ArrayLike, angle_flown: float) -> bool:
        _ = angle_flown
        return waypoint_reached(position, self.target, self.half_plane_normal, self.capture_radius)


@dataclass
class OrbitLeg(MissionLeg):
    """A circular leg, either a fixed sweep or an indefinite loiter."""

    orbit: OrbitPath
    sweep: float | None = None
    """Angle [rad] to fly before the leg ends. ``None`` loiters forever."""
    label: str = "orbit"

    def __post_init__(self) -> None:
        if self.sweep is not None:
            self.sweep = float(self.sweep)
            if self.sweep <= 0.0:
                raise GuidanceError(f"orbit sweep must be positive, got {self.sweep}")

    @property
    def path(self) -> OrbitPath:
        return self.orbit

    def command(self, position: ArrayLike, gains: GuidanceGains) -> GuidanceOutput:
        return self.orbit.command(position, gains)

    def is_complete(self, position: ArrayLike, angle_flown: float) -> bool:
        _ = position
        return self.sweep is not None and angle_flown >= self.sweep


# ── plan builders ─────────────────────────────────────────────────────────


def _fillet_radius(turn: float, turn_radius: float) -> float:
    """Distance [m] before a corner at which the turn has to start.

    The fillet of Beard & McLain section 11.2: an arc of radius ``R``
    tangent to both legs leaves the first one ``R tan(delta/2)`` short of
    the waypoint, for a turn of ``delta``. Using it as the acceptance
    radius means the aircraft hands over to the next leg exactly where the
    fillet would have begun.
    """
    if turn <= _EPS:
        return 0.0
    return float(min(turn_radius * np.tan(turn / 2.0), _MAX_FILLET_RADII * turn_radius))


def waypoint_plan(
    waypoints: Sequence[ArrayLike],
    *,
    airspeed: float,
    gains: GuidanceGains,
    loop: bool = False,
) -> list[MissionLeg]:
    """Straight legs through a waypoint list.

    Parameters
    ----------
    waypoints
        Two or more 3-vectors [m], world ENU. Altitude is interpolated
        along each leg, so a list at different altitudes flies a ramp
        rather than stepping at each waypoint.
    airspeed
        Commanded true airspeed [m/s] for every leg.
    gains
        Guidance design parameters; the turn radius they imply sets each
        waypoint's acceptance radius.
    loop
        Close the circuit by adding a leg from the last waypoint back to
        the first. The corner geometry at both ends then accounts for the
        closing leg, so a looped circuit has no discontinuity at the seam.
    """
    points = [np.asarray(w, dtype=float).reshape(3) for w in waypoints]
    if len(points) < 2:
        raise GuidanceError(f"a waypoint mission needs at least two waypoints, got {len(points)}")

    ordered = points + [points[0]] if loop else points
    directions = []
    for start, end in zip(ordered[:-1], ordered[1:], strict=True):
        if float(np.linalg.norm((end - start)[:2])) < _EPS:
            raise GuidanceError(
                "consecutive waypoints are at the same horizontal position — a fixed "
                "wing cannot fly a leg of zero ground length"
            )
        directions.append(end - start)

    turn_radius = gains.turn_radius(airspeed)
    legs: list[MissionLeg] = []
    for index, (start, direction) in enumerate(zip(ordered[:-1], directions, strict=True)):
        target = ordered[index + 1]
        is_last = index == len(directions) - 1
        outgoing = (
            directions[(index + 1) % len(directions)] if (loop or not is_last) else direction
        )
        turn = _turn_angle(direction, outgoing)
        legs.append(
            LineLeg(
                line=LinePath(origin=start, direction=direction, airspeed=airspeed),
                target=target,
                half_plane_normal=_bisector_normal(direction, outgoing),
                # The final waypoint of an open mission gets no fillet: it
                # is an arrival, not a corner, and the half-plane through it
                # is exactly the "abeam the waypoint" test.
                capture_radius=_fillet_radius(turn, turn_radius),
                label=f"leg {index + 1}",
            )
        )
    return legs


def orbit_plan(
    centre: ArrayLike,
    *,
    radius: float,
    airspeed: float,
    gains: GuidanceGains,
    direction: OrbitDirection = OrbitDirection.COUNTER_CLOCKWISE,
    turns: float | None = None,
) -> list[MissionLeg]:
    """A single orbit, for ``turns`` revolutions or indefinitely.

    Raises :class:`~uav_sim.guidance.fixed_wing_paths.GuidanceError` if the
    radius is tighter than the airframe can hold at ``airspeed``.
    """
    orbit = OrbitPath(centre=centre, radius=radius, direction=direction, airspeed=airspeed)
    orbit.validate(gains)
    sweep = None if turns is None else 2.0 * np.pi * float(turns)
    return [OrbitLeg(orbit=orbit, sweep=sweep, label="loiter" if turns is None else "orbit")]


def racetrack_plan(
    centre: ArrayLike,
    *,
    length: float,
    radius: float,
    heading: float,
    airspeed: float,
    gains: GuidanceGains,
    direction: OrbitDirection = OrbitDirection.COUNTER_CLOCKWISE,
) -> list[MissionLeg]:
    """A racetrack: two straight legs joined by half-orbits.

    Parameters
    ----------
    centre
        Centre [m] of the pattern, world ENU; its ``z`` is the altitude of
        the whole circuit.
    length
        Distance [m] between the two turn centres — the length of each
        straight leg.
    radius
        Turn radius [m] of the half-orbits. The straight legs are
        ``2 * radius`` apart.
    heading
        Course [rad] of the first straight leg, ENU.
    direction
        Which way round the circuit is flown.

    The four legs are built so that each straight arrives exactly tangent
    to the half-orbit that follows it. That is what closes the pattern: the
    exit of the second half-orbit is the entry of the first straight, with
    the same position *and* the same course, so the plan can be looped
    indefinitely without a discontinuity at the seam.

    Returns the four legs in flight order; pass them to
    :class:`FixedWingMission` with ``loop=True`` to fly the circuit
    repeatedly.
    """
    centre_arr = np.asarray(centre, dtype=float).reshape(3)
    if length <= 0.0:
        raise GuidanceError(f"racetrack length must be positive, got {length}")

    along = np.array([np.cos(heading), np.sin(heading), 0.0])
    # Rotating the leg direction +90 deg in ENU points to its left.
    left = np.array([-along[1], along[0], 0.0])
    sign = int(direction)

    far_centre = centre_arr + 0.5 * length * along
    near_centre = centre_arr - 0.5 * length * along
    # A counter-clockwise circuit flies its outbound leg on the right of
    # the centreline, so that arriving at the far turn centre's abeam point
    # it is already tangent to a left-hand turn. Clockwise mirrors it.
    outbound_offset = -sign * radius * left
    inbound_offset = sign * radius * left

    outbound_start = near_centre + outbound_offset
    outbound_end = far_centre + outbound_offset
    inbound_start = far_centre + inbound_offset
    inbound_end = near_centre + inbound_offset

    far_orbit = OrbitPath(centre=far_centre, radius=radius, direction=direction, airspeed=airspeed)
    near_orbit = OrbitPath(
        centre=near_centre, radius=radius, direction=direction, airspeed=airspeed
    )
    far_orbit.validate(gains)
    near_orbit.validate(gains)

    return [
        LineLeg(
            line=LinePath(origin=outbound_start, direction=along, airspeed=airspeed),
            target=outbound_end,
            # The half-orbit leaves the straight tangentially, so the
            # "outgoing direction" at the hand-over is the straight's own
            # direction and the bisector degenerates to it.
            half_plane_normal=along,
            capture_radius=0.0,
            label="outbound",
        ),
        OrbitLeg(orbit=far_orbit, sweep=np.pi, label="far turn"),
        LineLeg(
            line=LinePath(origin=inbound_start, direction=-along, airspeed=airspeed),
            target=inbound_end,
            half_plane_normal=-along,
            capture_radius=0.0,
            label="inbound",
        ),
        OrbitLeg(orbit=near_orbit, sweep=np.pi, label="near turn"),
    ]


def return_to_launch_plan(
    position: ArrayLike,
    home: ArrayLike,
    *,
    safe_altitude: float,
    airspeed: float,
    gains: GuidanceGains,
    loiter_radius: float | None = None,
    direction: OrbitDirection = OrbitDirection.COUNTER_CLOCKWISE,
) -> list[MissionLeg]:
    """Climb to a safe altitude, fly home, and loiter there.

    Parameters
    ----------
    position
        Where the aircraft is now [m], world ENU.
    home
        Launch point [m], world ENU.
    safe_altitude
        Altitude [m] to transit at. There is deliberately no default: the
        clearance a return needs is a property of the site, not of the
        aircraft, and a made-up number here would be a made-up terrain
        assumption.
    loiter_radius
        Radius [m] of the holding orbit. Defaults to twice the bank-limited
        turn radius, which holds the circle at about 27 deg of bank instead
        of sitting on the roll limit with nothing left for a disturbance.

    The climb is commanded from the first instant rather than as a separate
    phase: the transit leg is level *at* ``safe_altitude``, so the altitude
    loop starts climbing immediately and the aircraft is at height long
    before it arrives. A fixed wing cannot climb in place, so a distinct
    "climb first, then turn for home" phase would only mean climbing in the
    wrong direction.

    The transit leg hands over to the loiter when the aircraft first
    reaches the holding circle, so it enters the orbit from the rim rather
    than flying over the centre. If it is already inside that circle there
    is no transit leg at all — the orbit vector field commands straight
    outward from the centre and spirals onto the radius by itself.
    """
    position_arr = np.asarray(position, dtype=float).reshape(3)
    home_arr = np.asarray(home, dtype=float).reshape(3)
    radius = loiter_radius if loiter_radius is not None else 2.0 * gains.turn_radius(airspeed)

    hold_centre = np.array([home_arr[0], home_arr[1], float(safe_altitude)])
    orbit = OrbitPath(centre=hold_centre, radius=radius, direction=direction, airspeed=airspeed)
    orbit.validate(gains)
    loiter = OrbitLeg(orbit=orbit, sweep=None, label="loiter")

    transit_start = np.array([position_arr[0], position_arr[1], float(safe_altitude)])
    to_home = (hold_centre - transit_start)[:2]
    if float(np.linalg.norm(to_home)) <= radius:
        return [loiter]

    return [
        LineLeg(
            line=LinePath(
                origin=transit_start, direction=hold_centre - transit_start, airspeed=airspeed
            ),
            target=hold_centre,
            half_plane_normal=hold_centre - transit_start,
            capture_radius=radius,
            label="transit home",
        ),
        loiter,
    ]


# ── the mission manager ───────────────────────────────────────────────────


@dataclass
class MissionDiagnostics:
    """What the mission is doing right now, for plots and assertions."""

    leg_index: int = 0
    leg_label: str = ""
    path_error: float = 0.0
    """Signed distance from the active path [m]; see
    :attr:`~uav_sim.guidance.fixed_wing_paths.GuidanceOutput.path_error`."""
    course_command: float = 0.0
    altitude_command: float = 0.0
    angle_flown: float = 0.0
    """Angle [rad] swept around the active orbit, zero on a straight leg."""
    laps: int = 0
    complete: bool = False


class FixedWingMission:
    """Turns a list of legs into a stream of autopilot commands.

    The manager owns exactly one thing the paths do not: *which leg we are
    on*. Everything else — where the paths are, what course they ask for —
    is stateless geometry it delegates to.

    Parameters
    ----------
    params
        Airframe parameters, used for the default airspeed and gravity.
    legs
        The mission, in flight order. May be empty and set later with
        :meth:`fly`.
    gains
        Guidance design parameters. Defaults to :class:`GuidanceGains` with
        the airframe's gravity.
    loop
        Restart at the first leg when the last one completes, incrementing
        :attr:`MissionDiagnostics.laps`.
    home
        Launch point [m] for :meth:`return_to_launch`. Defaults to the
        first waypoint of ``legs``.

    Examples
    --------
    >>> import numpy as np
    >>> from uav_sim.guidance import FixedWingMission, waypoint_plan
    >>> from uav_sim.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing
    >>> aircraft = create_fixed_wing(FixedWingPreset.MINI_TRAINER)
    >>> _ = aircraft.reset_trimmed(airspeed=12.0, altitude=60.0)
    >>> mission = FixedWingMission(aircraft.fw_params)
    >>> mission.fly(waypoint_plan(
    ...     [[0.0, 0.0, 60.0], [400.0, 0.0, 60.0], [400.0, 400.0, 80.0]],
    ...     airspeed=12.0, gains=mission.gains))
    >>> command = mission.update(aircraft.state)
    >>> command.airspeed
    12.0
    >>> mission.diagnostics.leg_label
    'leg 1'
    """

    def __init__(
        self,
        params: FixedWingParams,
        legs: Sequence[MissionLeg] = (),
        gains: GuidanceGains | None = None,
        *,
        loop: bool = False,
        home: ArrayLike | None = None,
    ) -> None:
        self.params = params
        self.gains = gains or GuidanceGains(gravity=params.gravity)
        self._legs: list[MissionLeg] = list(legs)
        self.loop = bool(loop)
        self._home = None if home is None else np.asarray(home, dtype=float).reshape(3).copy()
        self.diagnostics = MissionDiagnostics()
        self._reset_progress()

    # ── plan management ───────────────────────────────────────────────────

    @property
    def legs(self) -> tuple[MissionLeg, ...]:
        """The legs currently loaded, in flight order."""
        return tuple(self._legs)

    @property
    def home(self) -> NDArray[np.floating] | None:
        """Launch point [m], or ``None`` if the mission has no waypoints."""
        if self._home is not None:
            return self._home.copy()
        for leg in self._legs:
            if isinstance(leg, LineLeg):
                return leg.line.origin.copy()
        return None

    @property
    def is_complete(self) -> bool:
        """True once a non-looping mission has finished its last leg."""
        return self.diagnostics.complete

    def fly(self, legs: Sequence[MissionLeg], *, loop: bool | None = None) -> None:
        """Load a new plan and start it from the first leg."""
        legs = list(legs)
        if not legs:
            raise GuidanceError("a mission needs at least one leg")
        self._legs = legs
        if loop is not None:
            self.loop = bool(loop)
        self._reset_progress()

    def return_to_launch(
        self,
        state: ArrayLike,
        *,
        safe_altitude: float,
        airspeed: float | None = None,
        loiter_radius: float | None = None,
        direction: OrbitDirection = OrbitDirection.COUNTER_CLOCKWISE,
    ) -> None:
        """Abandon the current plan and fly :func:`return_to_launch_plan`.

        Callable at any point in a mission — that is the whole point of it
        — and it takes the current state rather than assuming the aircraft
        is where the plan expected it to be.
        """
        home = self.home
        if home is None:
            raise GuidanceError(
                "return-to-launch needs a home position: pass `home=` to the mission "
                "or load a plan containing at least one straight leg"
            )
        position = np.asarray(state, dtype=float).reshape(-1)[:3]
        self.fly(
            return_to_launch_plan(
                position,
                home,
                safe_altitude=safe_altitude,
                airspeed=airspeed if airspeed is not None else self.params.cruise_airspeed,
                gains=self.gains,
                loiter_radius=loiter_radius,
                direction=direction,
            ),
            loop=False,
        )

    def reset(self) -> None:
        """Rewind to the first leg without changing the plan."""
        self._reset_progress()

    def _reset_progress(self) -> None:
        self._index = 0
        self._angle_flown = 0.0
        self._last_angle: float | None = None
        self._laps = 0
        self.diagnostics = MissionDiagnostics(
            leg_label=self._legs[0].label if self._legs else "",
        )

    # ── execution ─────────────────────────────────────────────────────────

    def update(self, state: ArrayLike) -> AutopilotCommand:
        """Advance the mission and return the command to hold now.

        Parameters
        ----------
        state
            Aircraft state vector (12 elements, FLU/ENU); only the position
            block is read.

        Once a non-looping mission runs out of legs it keeps flying the
        last one and reports ``complete``. Holding the final course beats
        the alternatives — freezing the command leaves an aircraft banked,
        and there is nothing else it could safely be told to do without
        being given a new plan.
        """
        if not self._legs:
            raise GuidanceError("mission has no legs — call fly() with a plan first")
        position = np.asarray(state, dtype=float).reshape(-1)[:3]

        self._accumulate_orbit_angle(position)
        self._advance(position)

        leg = self._legs[self._index]
        output = leg.command(position, self.gains)
        self.diagnostics = MissionDiagnostics(
            leg_index=self._index,
            leg_label=leg.label,
            path_error=output.path_error,
            course_command=output.course,
            altitude_command=output.altitude,
            angle_flown=self._angle_flown,
            laps=self._laps,
            complete=self.diagnostics.complete,
        )
        return AutopilotCommand(
            altitude=output.altitude,
            airspeed=output.airspeed,
            course=output.course,
            roll_feedforward=output.roll_feedforward,
        )

    def _accumulate_orbit_angle(self, position: NDArray[np.floating]) -> None:
        """Integrate the polar angle swept around the active orbit.

        Counting swept angle rather than testing "am I back where I
        started" is what makes a half-orbit terminate exactly once. The
        increment is wrapped to ``(-pi, pi]`` before accumulating, so it
        stays correct across the +-pi branch cut and cannot jump a lap.
        """
        leg = self._legs[self._index]
        if not isinstance(leg, OrbitLeg):
            self._angle_flown = 0.0
            self._last_angle = None
            return
        if leg.orbit.radial_distance(position) < _EPS:
            # Directly over the centre the polar angle is undefined; hold
            # the last one rather than accumulating noise.
            return
        angle = leg.orbit.angle_at(position)
        if self._last_angle is not None:
            delta = float(
                np.arctan2(np.sin(angle - self._last_angle), np.cos(angle - self._last_angle))
            )
            self._angle_flown += int(leg.orbit.direction) * delta
        self._last_angle = angle

    def _advance(self, position: NDArray[np.floating]) -> None:
        """Step past every leg that is finished at this position."""
        # Bounded rather than `while True`: a degenerate plan whose legs are
        # all instantly complete would otherwise spin forever.
        for _ in range(len(self._legs) + 1):
            if not self._legs[self._index].is_complete(position, self._angle_flown):
                return
            if self._index + 1 < len(self._legs):
                self._enter(self._index + 1, position)
            elif self.loop:
                self._laps += 1
                self._enter(0, position)
            else:
                self.diagnostics.complete = True
                return

    def _enter(self, index: int, position: NDArray[np.floating]) -> None:
        self._index = index
        self._angle_flown = 0.0
        leg = self._legs[index]
        self._last_angle = (
            leg.orbit.angle_at(position)
            if isinstance(leg, OrbitLeg) and leg.orbit.radial_distance(position) > _EPS
            else None
        )
