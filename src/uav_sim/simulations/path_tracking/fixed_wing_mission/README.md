# Fixed-Wing Mission Navigation

Waypoint legs, a racetrack pattern and a return-to-launch, flown in one
flight by a guidance layer sitting on top of the fixed-wing autopilot.

The guidance follows *paths*, not points. Each leg supplies a commanded
course from a vector field, so the aircraft converges onto the line and
stays on it rather than converging onto a heading that happens to point at
the next waypoint.

## Key equations

Straight-line following (Beard & McLain eq. 10.8), with the cross-track
error `e` measured positive to the **left** of the path in this library's
ENU frame:

$$\chi^c = \chi_q - \chi_\infty \frac{2}{\pi} \arctan(k_{path}\, e)$$

Orbit following (eq. 10.13), with `\lambda = +1` counter-clockwise in ENU:

$$\chi^c = \theta + \lambda\left[\frac{\pi}{2}
          + \arctan\left(k_{orbit}\frac{d - \rho}{\rho}\right)\right]$$

Waypoint acceptance is the half-plane test of eq. 11.1 **or** a capture
radius taken from the fillet geometry of section 11.2, whichever fires
first. The half-plane is what stops an aircraft that overshoots a waypoint
circling it forever.

## Reference

R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and Practice,"
Princeton University Press, 2012 — Chapters 10 and 11.

## Usage

```bash
flybots run fixed_wing_mission
# or
python -m uav_sim.simulations.path_tracking.fixed_wing_mission
```

## Result

![fixed_wing_mission](fixed_wing_mission.gif)

Running the simulation rewrites that GIF and a `fixed_wing_mission_log.json`
alongside it. Measured on the mini trainer, once each phase has settled:

| Phase | Capture peak | Settled mean | Settled max |
|---|---|---|---|
| Waypoints | 12.8 m | 6.1 m | 14.3 m |
| Racetrack | 216.0 m | 3.8 m | 14.1 m |
| Return to launch | 35.4 m | 2.0 m | 6.7 m |

The capture peaks are the point of the demo rather than a failing: each
mode change happens wherever the aircraft happens to be, and the settled
figures are what the guidance converges to from there. The waypoint
"error" of up to 14 m is mostly deliberate corner-cutting — the acceptance
radius hands over to the next leg one fillet radius before the waypoint, so
the aircraft is genuinely off the old line while turning onto the new one.
