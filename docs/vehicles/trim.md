# Trim and equilibrium

Trim is the attitude and control setting at which steady flight is an
**equilibrium** — the aircraft holds airspeed, altitude and attitude with
no further intervention.

It matters for two reasons. Starting a simulation from trim removes the
pitch transient that otherwise dominates the first few seconds, and the
trim controls are the feed-forward term any sensible autopilot is built
around.

> Beard & McLain, *Small Unmanned Aircraft: Theory and Practice*, Chapter 5.

## Solving for it

```python
from uav_sim.vehicles.fixed_wing import compute_trim, get_fixed_wing_params, FixedWingPreset

params = get_fixed_wing_params(FixedWingPreset.AEROSONDE)
trim = compute_trim(params, airspeed=35.0)

trim.alpha_deg      #  0.85 deg
trim.elevator       # -0.027 rad
trim.throttle       #  0.464
trim.residual       #  1.4e-11
trim.controls       # [elevator, aileron, rudder, throttle] — ready to step()
```

Or in one call on the aircraft:

```python
aircraft = create_fixed_wing(FixedWingPreset.AEROSONDE)
controls = aircraft.reset_trimmed(airspeed=35.0, altitude=200.0)
```

## What is being solved

Three unknowns — angle of attack, elevator, throttle — against three
equilibrium conditions. In steady flight the accelerations vanish, so the
applied forces exactly cancel the body-frame gravity components, and the
pitching moment is zero:

$$
\begin{aligned}
\frac{f_x}{m} - g\sin\theta &= 0 \\
\frac{f_z}{m} + g\cos\theta &= 0 \\
m_{pitch} &= 0
\end{aligned}
$$

with $\theta = \alpha + \gamma$ for a flight path angle $\gamma$. It is
solved numerically with `scipy.optimize.least_squares` from several
starting points, because the post-stall region of the lift curve is flat
enough to strand a solver seeded badly.

`residual` is the norm of that system at the solution. Anything below
`1e-3` is a genuine trim point; the presets all solve to around `1e-11`.

## Reading the envelope

```bash
uav-sim trim aerosonde
```

```text
Va m/s  alpha deg  elev deg   throttle
  13.2      14.83    -30.00      0.335
  17.6       7.13    -19.20      0.302
  22.0       4.06    -10.45      0.335
  26.4       2.44     -5.44      0.386
  30.8       1.48     -2.32      0.447
  35.2       0.85     -1.56      0.464
  39.6       0.06      0.61      0.529
```

Three physically-correct trends fall out:

- **Alpha decreases with airspeed.** Faster flight needs less incidence for
  the same lift, since lift goes as $V_a^2$.
- **Elevator moves from strongly negative toward positive.** Holding a high
  incidence at low speed takes a lot of up-elevator.
- **Throttle has a minimum in the middle.** This is the classic
  power-required curve: induced drag dominates at low speed, parasitic drag
  at high speed, and the cheapest cruise sits between them.

## Climbing trim

```python
trim = compute_trim(params, airspeed=35.0, climb_rate=2.0)
```

Climbing needs more throttle than level flight at the same airspeed, since
the propeller now supplies the potential-energy rate as well as drag. There
is a test asserting exactly that.

## When there is no solution

`compute_trim` raises `TrimError` rather than returning a plausible-looking
answer that is not an equilibrium:

```python
from uav_sim.vehicles.fixed_wing import TrimError

try:
    compute_trim(params, airspeed=5.0)      # well below stall
except TrimError as error:
    print(error)
    # No trim found at 5.0 m/s with +0.0 m/s climb (residual 0.42).
    # The airspeed is likely below stall (~12.5 m/s) or the climb rate
    # is too steep.
```

Common causes:

- **Below stall.** The wing cannot make enough lift at any incidence.
- **Climb rate beyond the excess thrust available.**
- **Climb rate exceeding airspeed**, which makes the flight path angle
  undefined — rejected up front.

## Using trim as a feed-forward

This is how `FixedWingAutopilot` uses it:

```python
class FixedWingAutopilot:
    def _solve_trim_throttle(self) -> float:
        try:
            return compute_trim(self.params, airspeed=self.params.cruise_airspeed).throttle
        except TrimError:
            return 0.5
```

The airspeed loop then only has to supply the *correction* around trim
throttle, not discover the whole operating point through its integrator.
The same idea appears in the [VTOL controller](/vehicles/vtol), which feeds
forward the incidence the wing needs to carry the aircraft.

## Trim as an acceptance test

Open-loop flight from trim is the strongest single check on a flight model.
If any force or moment is inconsistent — a sign error, a missing term, a
frame mismatch — trim is not an equilibrium and the aircraft drifts.

```python
aircraft = create_fixed_wing(preset)
controls = aircraft.reset_trimmed(altitude=300.0)
for _ in range(6000):
    aircraft.step(controls, 0.005)
assert aircraft.state[2] == pytest.approx(300.0, abs=1.0)
```

That test runs for all four presets. A model that passes it is internally
consistent in a way no shape assertion can establish.

## See also

- [Fixed-wing model](/vehicles/fixed-wing)
- [Airframe presets](/vehicles/presets)
- [Autopilots](/vehicles/autopilots)
