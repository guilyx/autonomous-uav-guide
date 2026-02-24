<!-- Erwin Lejeune — 2026-02-24 -->
# Flight Operations Demo

## Problem Statement

This article documents a mission-level execution chain where multiple algorithms are composed into a complete sortie:
arming, takeoff, offboard navigation, loiter, and landing.

## System Composition

The workflow combines:

- state-estimation updates for localization
- path planning for global route generation
- path tracking for local command execution
- state machine transitions for operation safety

## Mission-State Formulation

The behavior is modeled as a finite-state machine:

$$
S = \{ARM, TAKEOFF, OFFBOARD, LOITER, LAND\}
$$

Transition guards depend on altitude thresholds, waypoint completion, and velocity limits.

## Operational Procedure

1. Verify arming and estimator health checks.
2. Execute controlled takeoff to mission altitude.
3. Follow planned route in OFFBOARD mode.
4. Hold terminal position in LOITER.
5. Execute descent and disarm sequence.

## Failure Modes and Diagnostics

- Mode thrashing can occur if transition guards are too loose.
- Offboard timeout protection must override stale commands.
- Landing requires robust vertical-speed limits to avoid hard touchdown.

## Implementation and Execution

```bash
python -m uav_sim.simulations.path_tracking.flight_ops_demo
```

## Evidence

![Flight Ops Demo](https://media.githubusercontent.com/media/guilyx/autonomous-uav-guide/main/src/uav_sim/simulations/path_tracking/flight_ops_demo/flight_ops_demo.gif)

## References

- [PX4 Offboard Control Concepts](https://docs.px4.io/main/en/flight_modes/offboard.html)
- [Beard and McLain, Small Unmanned Aircraft: Theory and Practice](https://press.princeton.edu/books/hardcover/9780691149219/small-unmanned-aircraft)

## Related Algorithms

- [PID Hover](/simulations/path-tracking/pid-hover)
- [A* 3D](/simulations/path-planning/astar-3d)
- [Costmap Navigation](/simulations/environment/costmap-navigation)
