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
