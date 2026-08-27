<!-- Erwin Lejeune — 2026-02-24 -->
# VTOL Transition

## Problem Statement

VTOL transition combines multirotor-style hover control with fixed-wing forward-flight aerodynamics.
The key challenge is managing control authority handoff across flight regimes.

## Model and Formulation

Blend hover and forward-flight controllers with scheduling variable `\sigma \in [0,1]`:

$$
u = (1-\sigma)u_{hover} + \sigma u_{wing}
$$

with `\sigma` scheduled by airspeed, pitch, and altitude envelopes.

## Practical Notes

- **A partial tilt is not a transition.** Ramping the rotors to 30° never
  leaves rotor-borne flight: the wing is along for the ride and the demo is
  a slow quadrotor. The interesting regime is the handover itself, which
  only happens when the rotors go the full 90° and lift authority actually
  migrates to the wing. In cruise here the wing carries 95 % of the weight.
- **Draw the tilt, don't imply it.** This demo was previously rendered with
  a fixed-geometry quadrotor, which hid the one thing it exists to show: the
  rotors rotating through 90°. It now uses `draw_vtol_3d`, whose nacelles
  sit at the aircraft's actual tilt angle, so hover, transition and cruise
  are distinguishable from the picture alone.
- **Let the mode machine own the schedule.** `VTOLController` switches on
  measured airspeed against the wing's stall margin, so the tilt ramp is a
  consequence of the aircraft being ready rather than an open-loop timer
  that happens to work at one set of conditions.
- Watch altitude through the transition, not just at the ends. That is
  where a single altitude law has to degrade gracefully from pure rotor
  lift to pure wing lift; the atlas demo holds it to 4.8 m peak deviation.

- Transition corridors need explicit safety constraints.
- Inadequate gain scheduling causes pitch excursions and altitude loss.
- Propulsion and control-surface limits must be jointly managed.

## Evidence

![VTOL Transition](https://media.githubusercontent.com/media/guilyx/flybots/main/src/flybots/simulations/vehicles/vtol_transition/vtol_transition.gif)

## References

- [Lustosa et al., Dynamic Transition for VTOL UAVs](https://doi.org/10.1109/AERO.2019.8741928)
- [Sun et al., Review of Tilt-Rotor and VTOL Transition Control](https://doi.org/10.3390/aerospace9070390)

## Related Algorithms

- [Fixed-Wing Flight](/simulations/vehicles/fixed-wing-flight)
- [PID Hover](/simulations/path-tracking/pid-hover)
