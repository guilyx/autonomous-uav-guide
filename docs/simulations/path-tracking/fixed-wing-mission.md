<!-- Erwin Lejeune — 2026-08-18 -->
# Fixed-Wing Mission Navigation

## Problem Statement

The [autopilot](/vehicles/autopilots) will hold any course, altitude and
airspeed you give it. A mission has to decide *which* — continuously, from
a waypoint list, a loiter point or a pattern, and from wherever the
aircraft happens to be when the mode changes under it.

This simulation flies all three mission modes in one flight, on one
airframe, through one guidance layer:

1. **Waypoint legs** — four points and a climb.
2. **Racetrack** — two straights joined by half-orbits, flown twice.
3. **Return to launch** — triggered mid-pattern.

The hand-overs are the point. Each mode change happens somewhere the new
plan did not anticipate, which is exactly the situation an RTL has to cope
with for real.

## Model and Formulation

Straight-line following (Beard & McLain eq. 10.8), with cross-track error
`e` positive to the **left** of the path in this library's ENU frame:

$$\chi^c = \chi_q - \chi_\infty \frac{2}{\pi} \arctan(k_{path}\, e)$$

Orbit following (eq. 10.13), with $\lambda = +1$ counter-clockwise:

$$\chi^c = \theta + \lambda\left[\frac{\pi}{2}
          + \arctan\left(k_{orbit}\frac{d - \rho}{\rho}\right)\right]$$

The full derivation — including why both survive the NED→ENU conversion
unchanged, and how $k_{path}$ and $k_{orbit}$ are derived from the
autopilot's own course-loop bandwidth rather than tuned — is on the
[mission navigation](/vehicles/mission-navigation) page.

## Why the trainer

The 0.6 kg foam trainer cruises at 12 m/s and turns in under 15 m, so the
whole mission — a 900 m survey, a 200 m racetrack and a return — fits a
500 m world. The Aerosonde would need 125 m of turn radius and a world
eight times the size to show the same thing.

This is the same lesson as the [fixed-wing flight](/simulations/vehicles/fixed-wing-flight)
demo: the airframe has to match the world, or the demo is showing the
world's limits rather than the algorithm's.

## Algorithm Procedure

1. Build a plan: a list of legs, each a geometric path plus a termination
   test.
2. Each step, ask the active leg for a course, altitude and airspeed at the
   current position, and for the coordinated-turn bank its path requires.
3. Advance past any leg that is finished — a straight when the aircraft is
   inside the acceptance radius **or** past the half-plane, an orbit when
   it has swept its commanded angle.
4. Hand the result to the autopilot as an `AutopilotCommand`.

The manager owns exactly one piece of state: which leg is active.
Everything about where the paths are is stateless geometry.

## Measured Behaviour

| Phase | Capture peak | Settled mean | Settled max |
|---|---|---|---|
| Waypoints | 12.8 m | 6.1 m | 14.3 m |
| Racetrack | 216.0 m | 3.8 m | 14.1 m |
| Return to launch | 35.4 m | 2.0 m | 6.7 m |

"Settled" is measured 30 s after each mode change. The capture peaks are
the demo working, not failing: the racetrack's 216 m is simply how far
from the pattern the aircraft was when the survey finished, and the
guidance closes it.

The waypoint figures are the loosest of the three, and mostly by design —
the acceptance radius hands over to the next leg one fillet radius before
each corner, so the aircraft is genuinely off the old line while turning
onto the new one.

The flight ends 59.7 m from home at 146.2 m, against a commanded 60 m
holding circle at 145 m.

The error panel plots **magnitude on a log axis**, floored at 1 m. Each
mode change throws the error two orders of magnitude above where it
settles, and a linear axis wide enough to show a capture flattens
everything in between onto the zero line.

## Tuning Guidance

- `GuidanceGains.loop_separation` is the one knob that matters. Below
  about 1.5 the guidance asks for course changes faster than the autopilot
  delivers and the aircraft overshoots the line; well above 2 the capture
  is merely slow.
- `course_infinity` sets how hard the aircraft cuts in from far away. It
  must stay under 90°: at exactly 90° the aircraft is told to fly
  perpendicular to the path, which it can never roll out of in finite
  lateral distance.
- Pattern radii below the airframe's bank-limited turn radius are refused,
  not flown badly. If `racetrack_plan` raises, the pattern is the problem,
  not the gains.

## Usage

```bash
flybots run fixed_wing_mission
# or
python -m uav_sim.simulations.path_tracking.fixed_wing_mission
```

## Reference

R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and Practice*,
Princeton University Press, 2012 — Chapters 10 and 11.

## See also

- [Mission navigation](/vehicles/mission-navigation) — the guidance layer in full
- [Autopilots](/vehicles/autopilots) — the loop underneath it
- [Fixed-wing flight](/simulations/vehicles/fixed-wing-flight) — the autopilot alone
