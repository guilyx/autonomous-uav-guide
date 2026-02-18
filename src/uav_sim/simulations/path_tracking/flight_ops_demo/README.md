# Flight Operations Demo

Demonstrates high-level flight operations: takeoff, fly-to waypoints, loiter/hover, and landing. Shows the state machine transitions and how mission-level commands translate to controller setpoints.

## Key Equations

$$\text{DISARMED} \to \text{ARMED} \to \text{TAKEOFF} \to \text{HOVER} \to \text{OFFBOARD} \to \text{LAND}$$

## Reference

PX4 Autopilot Flight Modes Documentation. [Link](https://docs.px4.io/main/en/flight_modes/)

## Usage

```bash
python -m uav_sim.simulations.path_tracking.flight_ops_demo
```

## Result

![flight_ops_demo](flight_ops_demo.gif)
