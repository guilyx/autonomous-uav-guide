# Fixed-Wing Flight

Simulates a fixed-wing UAV using a simplified longitudinal/lateral dynamics model with lift, drag, and thrust. Demonstrates level flight, climb, and coordinated turns.

## Key Equations

$$L = \frac{1}{2}\rho v^2 S C_L, \quad D = \frac{1}{2}\rho v^2 S C_D$$

## Reference

R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and Practice," Princeton University Press, 2012.

## Usage

```bash
python -m uav_sim.simulations.vehicles.fixed_wing_flight
```

## Result

![fixed_wing_flight](fixed_wing_flight.gif)
