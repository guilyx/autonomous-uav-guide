# Airframe presets

```python
from flybots.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset

aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)
aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8, mass=4.0)   # override
```

## The four aircraft

| Preset | Mass | Span | AR | Wing loading | V stall | V cruise |
|---|---|---|---|---|---|---|
| `MINI_TRAINER` | 0.60 kg | 1.00 m | 5.56 | 32.7 N/m² | 6.3 m/s | 12 m/s |
| `SKYWALKER_X8` | 3.36 kg | 2.10 m | 5.88 | 44.0 N/m² | 8.4 m/s | 18 m/s |
| `AEROSONDE` | 13.5 kg | 2.90 m | 15.24 | 240.8 N/m² | 12.5 m/s | 35 m/s |
| `CARGO_UAV` | 25.0 kg | 4.00 m | 13.33 | 204.4 N/m² | 11.7 m/s | 32 m/s |

Stall speeds are **derived from each model's own lift curve**, not entered
by hand, so they cannot drift out of sync with the coefficients.

## Provenance

::: warning Only one of these is a published coefficient set
`AEROSONDE` uses the coefficients tabulated in Beard & McLain, Appendix E,
and is the reference airframe for this module.

The other three are **representative, not measured**. Their geometry and
mass come from the real aircraft; their aerodynamic coefficients are scaled
from the Aerosonde set with configuration-appropriate adjustments — a
flying wing gets reduced yaw stiffness and pitch damping because it has no
tail boom, a small foam model gets higher parasitic drag and an earlier,
softer stall.

They are good enough to fly, to tune controllers against, and to teach
with. They are not a substitute for wind-tunnel data on a specific
airframe, and you should not use them to predict how a real X8 behaves.
:::

## Choosing one

**`MINI_TRAINER`** — 0.6 kg foam model. Slow and docile, and the only
preset that fits comfortably in a world smaller than a few hundred metres.
Start here if you are building small demos.

**`SKYWALKER_X8`** — flying wing, no rudder. `Cndr` is zero, so the
autopilot's sideslip loop disables itself automatically. That makes it the
most interesting preset for control work: it cannot coordinate its own
turns, which is exactly the case that exposes a course loop closed on
heading instead of on course.

**`AEROSONDE`** — the textbook reference. Use it when you want numbers you
can check against Beard & McLain. Note the high aspect ratio and wing
loading: it is a long-endurance aircraft and needs room.

**`CARGO_UAV`** — 25 kg twin-boom. Heavy, high wing loading, sluggish. Good
for checking that a controller does not secretly depend on high control
authority.

## Custom airframes

```python
from dataclasses import replace
from flybots.vehicles.fixed_wing import (
    AeroCoefficients, FixedWing, FixedWingParams, PropulsionParams,
)
import numpy as np

aircraft = FixedWing(FixedWingParams(
    mass=2.0,
    wing_area=0.35,
    wing_span=1.6,
    chord=0.22,
    inertia=np.diag([0.06, 0.08, 0.12]),
    coeffs=replace(AeroCoefficients(), CLa=4.8, Cma=-1.5, Cmq=-18.0),
    propulsion=PropulsionParams(prop_area=0.02, k_motor=30.0),
    cruise_airspeed=16.0,
))
```

Or via the `CUSTOM` preset:

```python
aircraft = create_fixed_wing(FixedWingPreset.CUSTOM, mass=2.0, wing_area=0.35)
```

Check it trims before trusting it:

```python
from flybots.vehicles.fixed_wing import compute_trim
print(compute_trim(aircraft.fw_params, airspeed=16.0).residual)   # want < 1e-3
```

## Multirotor and VTOL presets

Multirotors have their own catalogue in `VehiclePreset`: four quadrotors
from a 27 g Crazyflie to a 3.6 kg Matrice, plus a `HEX_S550` hexacopter and
a coaxial `OCTO_X8`. See [Quadrotor](/vehicles/quadrotor) for the
four-rotor platforms and [Multirotor](/vehicles/multirotor) for the rest.

```python
from flybots.vehicles import VehiclePreset, create_multirotor

craft = create_multirotor(VehiclePreset.HEX_S550)
craft = create_multirotor(VehiclePreset.OCTO_X8, mass=5.2)   # override
```

The tilt-rotor currently ships a single default configuration; construct
`TiltrotorParams` directly to vary it.

## See also

- [Fixed-wing model](/vehicles/fixed-wing)
- [Multirotor model](/vehicles/multirotor)
- [Trim and equilibrium](/vehicles/trim)
