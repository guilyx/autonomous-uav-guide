# Tilt-Rotor VTOL Model

## Theory

A simplified tilt-rotor VTOL model that transitions between hover and cruise flight. The tilt angle linearly blends the thrust vector from vertical (hover) to horizontal (cruise) while engaging wing lift proportionally.

## Reference

R. Bapst et al., "Design and Implementation of an Unmanned Tail-Sitter," IROS, 2015.

## Simulation

```bash
uv run python simulations/vehicles/vtol_transition/run.py
```

![VTOL Transition](../../simulations/vehicles/vtol_transition/vtol_transition.gif)

## API

```python
from uav_sim.vehicles.vtol import Tiltrotor
vtol = Tiltrotor()
vtol.reset(state=initial_state)
# control: [total_thrust, tau_x, tau_y, tau_z, tilt_angle]
vtol.step(control, dt)
```
