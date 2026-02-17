# Fixed-Wing Aircraft Model

## Theory

A simplified 6DOF fixed-wing model using flat-earth assumptions. The aerodynamic forces (lift, drag) are modelled using stability derivatives, and the equations of motion follow the standard body-frame formulation with Euler angles.

$$L = \frac{1}{2} \rho V_a^2 S (C_{L_0} + C_{L_\alpha} \alpha)$$

$$D = \frac{1}{2} \rho V_a^2 S (C_{D_0} + C_{D_\alpha} \alpha^2)$$

## Reference

R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and Practice," Princeton University Press, 2012, Chapters 3-4.

## Simulation

```bash
uv run python simulations/vehicles/fixed_wing_flight/run.py
```

![Fixed-Wing Flight](../../simulations/vehicles/fixed_wing_flight/fixed_wing_flight.gif)

## API

```python
from uav_sim.vehicles.fixed_wing import FixedWing
fw = FixedWing()
fw.reset(state=initial_state)
fw.step(np.array([elevator, aileron, rudder, throttle]), dt)
```
