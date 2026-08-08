# Autopilots

## Fixed-wing

`FixedWingAutopilot` holds altitude, airspeed and course through a cascade
of nested loops.

> Beard & McLain, *Small Unmanned Aircraft: Theory and Practice*, Chapter 6.

```python
from uav_sim.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset
from uav_sim.control.fixed_wing_autopilot import FixedWingAutopilot, AutopilotCommand

aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)
aircraft.reset_trimmed(altitude=120.0)

pilot = FixedWingAutopilot(aircraft.fw_params)
command = AutopilotCommand(altitude=160.0, airspeed=20.0, course=1.0)

for _ in range(12_000):
    aircraft.step(pilot.compute(aircraft.state, command, 0.01), 0.01)
```

### Structure

```text
airspeed  ──PI───────────────────────────────▶ throttle
altitude  ──PI──▶ pitch cmd  ──PD───────────▶ elevator
course    ──PI──▶ roll cmd   ──PD───────────▶ aileron
sideslip  ──P────┐
yaw rate  ──washout──D──────────────────────▶ rudder
```

Inner loops run on attitude and are fast; outer loops run on trajectory and
are slow. That separation is what makes successive loop closure work
without a full multivariable synthesis.

### Gains are derived, not tuned

You do not supply gains. You supply an **actuator budget** — how much
deflection you are willing to spend on a given tracking error — and the
achievable bandwidth follows from the airframe's own control derivatives:

```python
kp_roll = sign(a_phi2) * max_surface / max_roll_error
omega_phi = sqrt(|a_phi2 * kp_roll|)
kd_roll = (2 * zeta * omega_phi - a_phi1) / a_phi2
```

where $a_{\phi_1}$ is the natural roll damping and $a_{\phi_2}$ the aileron
effectiveness, both computed from the coefficient set.

::: tip Why derive the bandwidth instead of requesting it
Asking for a specific bandwidth seems more direct, and it is what a first
implementation usually does. But if the requested bandwidth is *below* the
airframe's natural short-period or roll-subsidence frequency, the design
equations solve for gains that **remove** the stiffness and damping the
aircraft already has. The signs flip, rate feedback becomes positive
feedback, and the loop destabilises — on precisely the airframes that were
easiest to fly to begin with.

Deriving the bandwidth from the actuator budget cannot produce that,
because the resulting frequency is always at or above what the airframe
already does.
:::

The derivative gains are additionally floored at zero in the stabilising
direction. If an airframe is already better damped than the design target,
the controller leaves it alone rather than actively cancelling that damping
and creating a noise amplifier.

Result: the same autopilot flies all four presets with no retuning.

| Preset | Altitude error | Airspeed error | Course error |
|---|---|---|---|
| `MINI_TRAINER` | +0.33 m | −0.00 m/s | −0.00° |
| `SKYWALKER_X8` | +0.06 m | −0.00 m/s | −0.00° |
| `AEROSONDE` | +0.00 m | −0.00 m/s | +0.00° |
| `CARGO_UAV` | +0.00 m | −0.00 m/s | +0.00° |

Commanded: climb 40 m, turn 60°, increase speed 10%. Settled values after
120 s.

### The course loop closes on course, not heading

This distinction is the difference between a controller that settles and
one that hunts forever.

**Heading** (ψ) is where the nose points. **Course** (χ) is the direction
the aircraft is actually travelling. They differ by the sideslip angle.

The plant model behind the loop — turn rate $= g\tan\phi / V_a$ — describes
the *velocity vector*. Closing the loop on heading instead pairs a
heading measurement with a course plant.

On an aircraft that can zero its own sideslip, the two coincide and you get
away with it. On a rudderless flying wing like the X8, you do not:

```text
Closing on heading:
   t   course    psi     beta
 20.0   37.75    2.82   -32.81
 40.0   45.46   26.52   -19.79
 60.0   57.14   92.39   +34.69     <- flight path is right, nose is not
 80.0   62.48   32.48   -29.83
100.0   61.34   83.46   +22.04
```

The *course* column converges to the commanded 60° and stays there — the
aircraft is flying exactly where it was told. The nose oscillates ±30°
around it indefinitely, because the loop keeps commanding bank to correct a
heading error that only sideslip can fix, and the aircraft has no rudder.

Closing on course instead: 0.00° steady-state error on every preset.

```python
velocity_world = euler_to_rotation(*state[3:6]) @ state[6:9]
course = np.arctan2(velocity_world[1], velocity_world[0])
```

### Yaw damper

The rudder carries two terms: a sideslip regulator and a washed-out yaw-rate
damper. The washout is what lets it damp Dutch roll without fighting the
steady yaw rate of a turn — a constant rate decays out of the filter within
a few time constants.

Both scale with `sign(Cndr)`, so the loop works on any airframe and
disables itself entirely on a rudderless one where `Cndr == 0`.

### Anti-windup

The PI helper uses conditional integration: the integrator only advances
while the unsaturated output is in range, and the stored value is
additionally clamped so its own contribution cannot exceed the output
range.

Clamping the raw integral — the usual shortcut — still lets a large `ki`
push a saturated command far past the limit and then take seconds to
unwind.

### Tuning

```python
from uav_sim.control.fixed_wing_autopilot import AutopilotGains

pilot = FixedWingAutopilot(params, AutopilotGains(
    max_roll_error=np.radians(8.0),     # tighter roll loop
    max_altitude_error=15.0,            # more aggressive altitude capture
    max_roll=np.radians(30.0),          # gentler bank limit
))
```

Read `max_*_error` as "the error at which this loop saturates its
actuator". Smaller is tighter.

## VTOL

`VTOLController` handles the hover / transition / cruise / back-transition
sequence. Its design is covered on the [VTOL page](/vehicles/vtol) — the
short version is that control allocation flips through the transition,
blended on both airspeed and actual rotor tilt, with a trim-incidence
feed-forward and cancellation of the wing's own pitching moment.

## Multirotor

The cascaded stack in `uav_sim.control`:

```text
PositionController -> VelocityController -> AttitudeController -> RateController
```

composed by `FlightController`, with `StateManager` running
ARM → TAKEOFF → HOVER → TRACKING → LAND. Alternative multirotor tracking
controllers live in `uav_sim.path_tracking`: PID, LQR, MPC, pure pursuit,
and a geometric SO(3) controller.

## See also

- [Trim and equilibrium](/vehicles/trim)
- [Frames and conventions](/guide/conventions)
