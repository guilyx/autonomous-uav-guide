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

- Transition corridors need explicit safety constraints.
- Inadequate gain scheduling causes pitch excursions and altitude loss.
- Propulsion and control-surface limits must be jointly managed.

## Evidence

![VTOL Transition](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/vtol_transition/vtol_transition.gif)

## References

- [Lustosa et al., Dynamic Transition for VTOL UAVs](https://doi.org/10.1109/AERO.2019.8741928)
- [Sun et al., Review of Tilt-Rotor and VTOL Transition Control](https://doi.org/10.3390/aerospace9070390)

## Related Algorithms

- [Fixed-Wing Flight](/simulations/vehicles/fixed-wing-flight)
- [PID Hover](/simulations/path-tracking/pid-hover)
