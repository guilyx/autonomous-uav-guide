# Fixed-wing

A six-degree-of-freedom aircraft with the full non-linear aerodynamic
coefficient build-up from

> R. W. Beard, T. W. McLain, *Small Unmanned Aircraft: Theory and
> Practice*, Princeton University Press, 2012 — Chapter 4 and Appendix E.

Source: [`uav_sim/vehicles/fixed_wing/`](https://github.com/guilyx/flybots/tree/main/src/uav_sim/vehicles/fixed_wing)

## Quick start

```python
from uav_sim.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset

aircraft = create_fixed_wing(FixedWingPreset.AEROSONDE)
controls = aircraft.reset_trimmed(airspeed=35.0, altitude=200.0)

for _ in range(6000):
    aircraft.step(controls, 0.005)

print(aircraft.state[2])   # 200.0
print(aircraft.airspeed)   # 35.0
```

Thirty seconds of open-loop flight holding altitude to the metre. That is
the acceptance test for the whole model: if any force or moment is
inconsistent, trim is not an equilibrium and the aircraft wanders.

## State and control

```text
state   = [x, y, z, φ, θ, ψ, u, v, w, p, q, r]
control = [elevator, aileron, rudder, throttle]
```

Position is world **ENU**; velocity `[u, v, w]` is **body FLU**; rates are
body-frame. Surfaces are radians, clamped to ±30°; throttle is `[0, 1]`.

Storing velocity in the body frame is deliberate — angle of attack is
`atan2(-w, u)`, so the aerodynamics want it there and would otherwise
rotate it back on every evaluation.

## Derived quantities

```python
aircraft.airspeed            # true airspeed Va [m/s]
aircraft.alpha               # angle of attack [rad]
aircraft.beta                # sideslip [rad]
aircraft.pitch_up            # pitch, aerospace sign convention
aircraft.flight_path_angle   # climb angle [rad]
aircraft.load_factor         # lift / weight — the "g" being pulled
aircraft.is_stalled()        # |alpha| past the stall boundary
aircraft.velocity            # world-frame velocity
aircraft.last_wrench         # lift, drag, thrust from the last step
```

## The aerodynamic model

Forces and moments are built from dimensionless coefficients evaluated at
the current flow condition. The model is written in the textbook's
Forward-Right-Down frame and converted at the boundary — see
[Frames and conventions](/guide/conventions).

### Lift, with stall

$$
C_L(\alpha) = (1 - \sigma(\alpha))\,(C_{L_0} + C_{L_\alpha}\alpha)
            + \sigma(\alpha)\,\bigl(2\,\mathrm{sign}(\alpha)\sin^2\!\alpha\cos\alpha\bigr)
$$

$\sigma$ is a sigmoid that blends from the linear regime into flat-plate
behaviour past the stall angle. Without it, lift grows without bound and
the aircraft can "fly" at 60° of incidence.

| α | C_L |
|---|---|
| 0° | 0.23 |
| 10° | 1.21 |
| 20° | 2.18 |
| 27° | 1.58 |
| 45° | 0.71 |

Lift peaks near 20° and falls away — the aircraft can genuinely stall, and
recover.

### Drag, from the induced-drag polar

$$
C_D = C_{D_0} + C_{D_p} + \frac{(C_{L_0} + C_{L_\alpha}\alpha)^2}{\pi e AR}
    + C_{D_\alpha}\alpha^2
$$

This is where the Oswald efficiency $e$ and the wing span enter, through
the aspect ratio $AR = b^2/S$. A short, stubby wing pays more induced drag
than a long one at the same lift, which is why the trim throttle differs
between presets.

### Moments

$$
\begin{aligned}
m &= \bar q S c \left(C_{m_0} + C_{m_\alpha}\alpha
     + C_{m_q}\tfrac{c}{2V_a}q + C_{m_{\delta_e}}\delta_e\right) \\
l &= \bar q S b \left(C_{l_0} + C_{l_\beta}\beta + C_{l_p}\tfrac{b}{2V_a}p
     + C_{l_r}\tfrac{b}{2V_a}r + C_{l_{\delta_a}}\delta_a
     + C_{l_{\delta_r}}\delta_r\right) \\
n &= \bar q S b \left(C_{n_0} + C_{n_\beta}\beta + C_{n_p}\tfrac{b}{2V_a}p
     + C_{n_r}\tfrac{b}{2V_a}r + C_{n_{\delta_a}}\delta_a
     + C_{n_{\delta_r}}\delta_r\right)
\end{aligned}
$$

Three of these terms are what make the aircraft an *aircraft* rather than a
brick with a lift force:

| Derivative | Sign | What it does |
|---|---|---|
| $C_{m_\alpha}$ | negative | Static pitch stability. Pitch up, get a nose-down moment back. |
| $C_{m_q}$ | negative | Pitch damping. Without it the short-period mode never settles. |
| $C_{n_\beta}$ | positive | Weathercock stability. The nose swings into the relative wind. |
| $C_{l_p}$ | negative | Roll damping. |
| $C_{l_\beta}$ | negative | Dihedral effect — sideslip produces roll. |

The rate terms are non-dimensionalised by $c/2V_a$ and $b/2V_a$. Those
denominators look like they blow up at low airspeed, but $\bar q$ carries
$V_a^2$, so each product is linear in $V_a$ and vanishes cleanly at rest.

### Propulsion

Momentum theory: the propeller accelerates air from $V_a$ to
$k_{motor}\delta_t$.

$$
T = \tfrac{1}{2}\rho S_{prop} C_{prop}\left((k_{motor}\delta_t)^2 - V_a^2\right)
$$

Thrust therefore *falls off with airspeed*, which is what makes level flight
settle at a finite speed for a given throttle instead of accelerating
forever.

## Verified behaviour

Each of these has a test in
[`tests/test_fixed_wing_aero.py`](https://github.com/guilyx/flybots/blob/main/tests/test_fixed_wing_aero.py):

```text
Pitch damping        1 rad/s disturbance: 2.92° swing -> 0.19° after 8 s
Static stability     +0.1 rad alpha -> +19.6 rad/s² restoring (nose-down)
Weathercock          8.1° sideslip -> 7.4 rad/s² yaw into the wind
Side force           8.1° sideslip -> 4.5 m/s² lateral acceleration
Stall                CL peaks at 2.18 near 20°, falls to 0.54 by 35°
Trim                 residual < 1e-11 across the speed envelope
Open-loop trim       200 m held to 0.00 m over 30 s
```

::: tip Every parameter is load-bearing
There is a parametrised test asserting that perturbing `Cm0`, `Cma` or
`e_oswald` changes the dynamics. It exists because the previous model
declared all three and read none of them — leaving an aircraft with no
static pitch stability and no induced drag that still looked complete in
the source.
:::

## Custom coefficients

```python
from dataclasses import replace
from uav_sim.vehicles.fixed_wing import AeroCoefficients, FixedWing, FixedWingParams

unstable = replace(AeroCoefficients(), Cma=-0.2)   # nearly neutral in pitch
aircraft = FixedWing(FixedWingParams(coeffs=unstable, mass=10.0))
```

Making an aircraft unstable is a good way to see what the derivatives buy
you. Set `Cmq = 0.0` and watch the short period ring forever.

## Envelope

```python
params = aircraft.fw_params
params.stall_airspeed        # derived from the model's own CL_max
params.aspect_ratio          # b^2 / S
params.wing_loading()        # N/m^2
params.max_lift_coefficient
```

`stall_airspeed` is computed as $\sqrt{2mg/(\rho S C_{L,max})}$ by sweeping
the actual lift curve, rather than being a hand-entered constant that can
drift out of sync with the coefficients.

## See also

- [Trim and equilibrium](/vehicles/trim) — solving for steady flight
- [Airframe presets](/vehicles/presets) — the four supplied aircraft
- [Autopilots](/vehicles/autopilots) — altitude, airspeed and course hold
- [Fixed-wing flight simulation](/simulations/vehicles/fixed-wing-flight)
