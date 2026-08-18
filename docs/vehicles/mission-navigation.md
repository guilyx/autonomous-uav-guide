<!-- Erwin Lejeune — 2026-08-18 -->
# Mission navigation

An autopilot answers *hold this course, altitude and airspeed*. A mission
answers *which course, altitude and airspeed should I be holding right
now*. `uav_sim.guidance` is the layer in between for fixed-wing aircraft:
straight-line and orbit vector fields, waypoint sequencing, racetrack
patterns and return-to-launch.

> Beard & McLain, *Small Unmanned Aircraft: Theory and Practice*,
> Princeton University Press, 2012 — Chapter 10 (path following) and
> Chapter 11 (path management).

Source:
[`uav_sim/guidance/`](https://github.com/guilyx/flybots/tree/main/src/uav_sim/guidance)

## Quick start

```python
from uav_sim.control.fixed_wing_autopilot import FixedWingAutopilot
from uav_sim.guidance import FixedWingMission, GuidanceGains, waypoint_plan
from uav_sim.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing

aircraft = create_fixed_wing(FixedWingPreset.MINI_TRAINER)
aircraft.reset_trimmed(airspeed=12.0, altitude=120.0)

pilot = FixedWingAutopilot(aircraft.fw_params)
gains = GuidanceGains(gravity=aircraft.fw_params.gravity)
mission = FixedWingMission(
    aircraft.fw_params,
    waypoint_plan(
        [[0, 0, 120], [400, 0, 120], [400, 400, 150]],
        airspeed=12.0,
        gains=gains,
    ),
    gains,
)

while not mission.is_complete:
    command = mission.update(aircraft.state)
    aircraft.step(pilot.compute(aircraft.state, command, 0.01), 0.01)
```

`mission.update()` returns an
[`AutopilotCommand`](/vehicles/autopilots). Nothing about the autopilot
changes: the guidance is a layer above it, not a replacement for it.

## Follow the path, not the point

The obvious way to fly a waypoint list is to point the nose at the next
waypoint. It is also the way that leaves a standing lateral offset,
because the law is satisfied the moment the nose is on the target — no
matter how far off the line the aircraft has drifted getting there.

A cross-track law is satisfied only when the aircraft is *on the line*.
Beard & McLain's straight-line field (eq. 10.8) commands

$$\chi^c = \chi_q - \chi_\infty \frac{2}{\pi} \arctan(k_{path}\, e)$$

where $e$ is the cross-track error and $\chi_\infty$ the intercept angle
the law saturates at far from the path. On the line it commands the path
course; far off it commands a fixed intercept and closes at
$V_g \sin\chi_\infty$.

The orbit field (eq. 10.13) is the same idea about a circle:

$$\chi^c = \theta + \lambda\left[\frac{\pi}{2}
          + \arctan\left(k_{orbit}\frac{d - \rho}{\rho}\right)\right]$$

with $\theta$ the aircraft's polar angle about the centre, $d$ its
distance from it, and $\lambda$ the direction of travel. On the circle
this is the tangent; far outside it points at the centre; at the centre it
points straight out.

## ENU: both laws survive the conversion unchanged

::: warning Read this before trusting a sign
The book is written in **NED**, where course increases clockwise. This
library is **ENU**, where it increases counter-clockwise. Every course
formula needs converting at the boundary — see
[Frames and conventions](/guide/conventions).
:::

Both laws come out algebraically identical after the conversion, because
each carries two sign flips that cancel:

| | Book (NED) | Here (ENU) |
|---|---|---|
| Straight line | $e_{py}$ positive to the **right** of the path | $e$ positive to the **left** |
| Orbit | $\lambda = +1$ is **clockwise** | $\lambda = +1$ is **counter-clockwise** |

Flipping to ENU flips the sense of course; measuring the cross-track error
positive to the left flips it back. Same for the orbit, where the
direction convention absorbs the handedness change. So
`LinePath.command` and `OrbitPath.command` use the book's expressions
verbatim — but the fact that they *can* is a result, not an accident, and
the tests pin both signs explicitly:

```python
def test_left_of_the_path_steers_right():
    east = LinePath.between([0, 0, 100], [500, 0, 100], airspeed=20.0)
    assert east.command([0, 40, 100], GuidanceGains()).course < 0.0
```

An aircraft with either sign flipped still converges — onto the wrong side,
in the wrong direction — and looks entirely plausible in a plot.

## Gains are derived, not tuned

Same philosophy as the [autopilot](/vehicles/autopilots), and by the same
argument: **loop separation**. This guidance is one loop outside the
autopilot's course hold, so it has to be slower than it.

Linearise the straight-line law near the path:

$$\dot e = V_g \sin(\chi - \chi_q)
        \approx -V_g \chi_\infty \frac{2}{\pi} k_{path}\, e$$

a first-order response with pole $\omega_e = 2 V_g \chi_\infty k_{path} /
\pi$. Set that to the autopilot's course-loop bandwidth divided by
`loop_separation`, solve for $k_{path}$, and the design is done. The
orbit gain falls out of the identical linearisation about the circle, so
one number sets both and a straight leg and an orbit converge at the same
rate.

The course-loop bandwidth is read back out of `AutopilotGains` rather than
copied, so retuning the autopilot retunes the guidance with it. It is
evaluated at the airspeed of the leg being flown, not at cruise: the
autopilot freezes `kp_course` at its design point, so an aircraft on a
slow leg genuinely has a faster course loop than it was designed for.

What comes out, at `loop_separation = 2` and $\chi_\infty = 60°$:

| Airframe | $V_a$ [m/s] | $R_{min}$ [m] | $\omega_\chi$ [rad/s] | $\omega_e$ [rad/s] | $1/k_{path}$ [m] | roll-out width [m] |
|---|---|---|---|---|---|---|
| Mini trainer | 12 | 14.7 | 0.307 | 0.153 | 52 | 7.3 |
| Skywalker X8 | 18 | 33.0 | 0.204 | 0.102 | 117 | 16.5 |
| Cargo UAV | 32 | 104.4 | 0.115 | 0.057 | 371 | 52.2 |
| Aerosonde | 35 | 124.9 | 0.105 | 0.053 | 444 | 62.4 |

$1/k_{path}$ is the cross-track error at which the law asks for half its
maximum intercept — the number to look at when judging whether a capture
will be tight or lazy. It scales from 52 m for a foam trainer to 444 m for
an Aerosonde without anyone typing either figure.

### Why `loop_separation = 2`

The geometric floor is the **roll-out width**: an aircraft crossing at
$\chi_\infty$ and turning at its bank limit sweeps
$R_{min}(1 - \cos\chi_\infty)$ of lateral distance merely rolling out onto
the line. No law can capture from closer, whatever its gains, and there is
a test asserting the derived transition distance clears it on every
preset. It does, by a factor of 7.1 — the same on all four, because both
quantities scale with the turn radius and the ratio is a property of the
design rather than of the airframe.

The floor alone is optimistic, because it assumes the course loop tracks
instantly. Capturing a line from five turn radii off:

| `loop_separation` | Worst overshoot (mini trainer) | as % of $R_{min}$ |
|---|---|---|
| 1.0 | 3.35 m | 23 % |
| 1.5 | 0.28 m | 1.9 % |
| 2.0 | none | 0 % |

Overshoot is gone by 1.5 on all four airframes. `2.0` keeps a third again
of margin on that boundary and costs about 1.5× the capture time.

### What it converges to

Starting five turn radii off the line, and measuring over the last 15 % of
the flight:

| Airframe | Initial offset | Settled cross-track | as % of $R_{min}$ |
|---|---|---|---|
| Mini trainer | 73 m | 0.06 m | 0.43 % |
| Skywalker X8 | 165 m | 0.19 m | 0.59 % |
| Cargo UAV | 522 m | 0.61 m | 0.58 % |
| Aerosonde | 624 m | 0.73 m | 0.58 % |

A settled orbit does better still — radial error under a millimetre on all
four — because the course loop's PI drives an integrator plant, which
tracks the constant course rate of a circle with zero steady-state error
once its integrator has charged.

## The turn feed-forward

*Once its integrator has charged* is the catch. On a circuit of
alternating straights and turns, the aircraft never gets that long: every
hand-over steps the required turn rate, the integrator starts again, and
the lag shows up as a bulge on each turn entry and an overshoot on each
exit. Longer legs do not help, because the problem is the transient, not
the settling.

So every guidance command carries the coordinated-turn bank its path
requires — zero on a straight, $-\lambda\arctan(V_a^2 / g\rho)$ on an
orbit — in `AutopilotCommand.roll_feedforward`, and the course PI is left
regulating only the error around it. Beard & McLain add the same term to
their orbit follower for the same reason.

Flying a racetrack of 2.5 turn radii on the mini trainer:

| | Worst-case path error | Steady end-of-leg error |
|---|---|---|
| Feedback only | 18.3 m | 15.0 m |
| With turn bank fed forward | 6.3 m | 2.4 m |

The field defaults to zero, and at zero the autopilot is bit-for-bit the
pure feedback design it has always been. The PI is handed whatever bank
authority the feed-forward has not spent, so its anti-windup still
measures the real remaining range, and the sum is clamped to the roll
limit.

## Waypoint acceptance: a radius *and* a half-plane

A leg ends when **either** test fires:

- the aircraft is within the **capture radius** of the waypoint, or
- it has crossed the **half-plane** through the waypoint whose normal
  bisects the incoming and outgoing legs (eq. 11.1).

Each alone is wrong in a different way.

A radius alone can be missed entirely. An aircraft pushed wide — by wind,
by a corner tighter than it can turn, by arriving too fast — never enters
the circle. It turns back, misses again on the other side, and orbits the
waypoint for the rest of the flight. This is the classic failure, and it
is why the half-plane exists: once the aircraft is past the plane it is
past the waypoint, full stop.

A half-plane alone corners badly. It only fires *at* the waypoint, so the
aircraft flies all the way to the point before starting to turn and bulges
outside every corner.

The capture radius comes from the fillet geometry of section 11.2: an arc
of radius $R$ tangent to both legs leaves the first one
$R\tan(\delta/2)$ short of the waypoint, for a turn of $\delta$. Using
that as the acceptance radius hands over exactly where the fillet would
have begun — one turn radius early at a right-angle corner, nothing at all
at a waypoint the aircraft flies straight through, and capped at three
turn radii where a near-reversal sends the tangent to infinity.

::: tip This is why the demo's waypoint error looks large
Up to 14 m of "cross-track error" on the waypoint phase is mostly
deliberate. The aircraft leaves the old line one fillet radius before the
corner, so it *is* off that line while turning onto the next one.
:::

The final waypoint of an open mission gets no fillet at all. It is an
arrival, not a corner, and the half-plane through it is exactly the
"abeam the waypoint" test.

## Racetrack

Two straight legs joined by half-orbits:

```python
from uav_sim.guidance import racetrack_plan, OrbitDirection

circuit = racetrack_plan(
    [250, 240, 110],          # centre; its z is the pattern altitude
    length=200.0,             # distance between the two turn centres
    radius=45.0,
    heading=np.radians(20.0),
    airspeed=12.0,
    gains=gains,
    direction=OrbitDirection.COUNTER_CLOCKWISE,
)
mission.fly(circuit, loop=True)
```

The construction places each straight so it arrives *tangent* to the
half-orbit that follows. That is what closes the pattern: the exit of the
second half-orbit is the entry of the first straight, at the same position
and the same course, so `loop=True` repeats it indefinitely with no
discontinuity at the seam. The test flies four laps and asserts that
successive laps begin within 5 % of the pattern's turn radius of each other — measured
by which leg the mission manager reports, not by elapsed time, because the
aircraft flies a slightly longer path than the ideal perimeter and a fixed
lap time would compare two different places and call the difference drift.

Half-orbits terminate on **swept angle**, accumulated from the polar angle
about the centre and wrapped to $(-\pi, \pi]$ before adding, so a turn ends
after exactly half a revolution and cannot jump a lap across the $\pm\pi$
branch cut.

An orbit tighter than the airframe's bank-limited turn radius is refused
rather than flown badly:

```python
GuidanceError: orbit radius 60.0 m is below the bank-limited turn radius
124.9 m at 35.0 m/s — holding it would need 61.5 deg of bank against a
45.0 deg limit
```

## Return to launch

```python
mission.return_to_launch(aircraft.state, safe_altitude=145.0, loiter_radius=60.0)
```

Callable at any point in a mission — that is the whole point of it — and
it takes the current state rather than assuming the aircraft is where the
plan expected it to be. It builds two legs: a straight transit home, and a
loiter orbit there.

Three decisions worth stating:

**The climb is commanded from the first instant**, not scheduled as a
separate phase. The transit leg is level *at* the safe altitude, so the
altitude loop starts climbing immediately and the aircraft is at height
long before it arrives. A fixed wing cannot climb in place, so a distinct
"climb first, then turn for home" phase would only mean climbing in the
wrong direction.

**`safe_altitude` has no default.** The clearance a return needs is a
property of the site, not of the aircraft, and a made-up number here would
be a made-up terrain assumption.

**The transit hands over at the holding circle**, so the aircraft enters
the loiter from the rim rather than flying over the centre. If it is
already inside that circle there is no transit leg at all — the orbit
field commands straight outward from the centre and spirals onto the
radius by itself.

## Wind

::: danger The airframe model has no wind
`uav_sim.vehicles.fixed_wing` has no wind field. Its body velocity is both
the air-relative and the inertial velocity, so nothing here is tested
against wind and no claim is made that it is wind-compensating. Adding
wind and gust models is a separate open [roadmap](/ROADMAP) item.
:::

What *is* true is that the layer is built so wind enters in one place when
it lands. The guidance closes on inertial cross-track distance and
commands **course**, not heading, and the autopilot behind it also tracks
course. A crabbing aircraft therefore already has the right feedback
structure: the nose points wherever it must for the *velocity vector* to
lie along the path.

The waypoint acceptance test is the other half of that. The half-plane is
what makes an overshoot recoverable — whether the overshoot came from wind,
from a corner tighter than the airframe can turn, or from arriving fast.
There is a test that flies a 150° reversal in a space far tighter than the
turn radius, so the aircraft cannot possibly make the corner, and asserts
the mission still advances and completes.

## Reference

| Piece | API |
|---|---|
| Straight-line field | `LinePath.command` |
| Orbit field | `OrbitPath.command` |
| Turn radius | `minimum_turn_radius`, `GuidanceGains.turn_radius` |
| Derived gains | `GuidanceGains.line_gain`, `.orbit_gain` |
| Acceptance test | `waypoint_reached` |
| Plans | `waypoint_plan`, `orbit_plan`, `racetrack_plan`, `return_to_launch_plan` |
| Execution | `FixedWingMission.update`, `.fly`, `.return_to_launch` |

## See also

- [Autopilots](/vehicles/autopilots) — the loop underneath this one
- [Fixed-wing model](/vehicles/fixed-wing) — the airframe being flown
- [Trim and equilibrium](/vehicles/trim) — where every flight starts
- [Frames and conventions](/guide/conventions) — why the signs are what they are
- [Fixed-wing mission simulation](/simulations/path-tracking/fixed-wing-mission)
