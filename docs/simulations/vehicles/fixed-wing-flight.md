<!-- Erwin Lejeune — 2026-02-24 -->
# Fixed-Wing Flight

## Problem Statement

Fixed-wing platforms have fundamentally different dynamics and control constraints compared to hovering multirotors.
This article captures aerodynamic lift-driven flight behavior for autonomous guidance studies.

## Model and Formulation

Longitudinal force balance:

$$
m\dot{V} = T\cos\alpha - D - mg\sin\gamma
$$

Lift relation:

$$
L=\frac{1}{2}\rho V^2 S C_L(\alpha)
$$

## Practical Notes

- **The airframe has to match the world.** A fixed-wing has a stall speed
  and a turn radius, and both scale with the aircraft. A 13.5 kg Aerosonde
  trims near 35 m/s and needs hundreds of metres to come round; asked to
  hold 8 m/s in a 30 m box it is below stall from the first frame and
  simply falls out of the sky. The demo flies a 0.6 kg trainer — 6.3 m/s
  stall, 12 m/s cruise — in a 200 m world, several turn diameters across.
- **Start from trim, not from a guess.** `reset_trimmed()` solves for the
  equilibrium first, so the aircraft begins balanced. An untrimmed start
  spends its first seconds porpoising, and on a short clip that transient
  is most of what you see — the controller never gets a fair showing.
- Check the achievable turn radius before laying out a circuit:
  `r = V² / (g tan φ)`. At 12 m/s and 45° of bank that is 14.7 m.

- Minimum airspeed constraints are safety-critical.
- Turn-rate limits depend on bank angle and available lift.
- Wind models strongly influence path-following performance.

## Evidence

![Fixed Wing Flight](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif)

## References

- [Stevens and Lewis, Aircraft Control and Simulation](https://www.wiley.com/en-us/Aircraft+Control+and+Simulation%2C+3rd+Edition-p-9781119174882)
- [Beard and McLain, Small Unmanned Aircraft](https://press.princeton.edu/books/hardcover/9780691149219/small-unmanned-aircraft)

## Related Algorithms

- [VTOL Transition](/simulations/vehicles/vtol-transition)
- [Frenet Optimal](/simulations/trajectory-planning/frenet-optimal)
