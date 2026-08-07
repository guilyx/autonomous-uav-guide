---
layout: home

hero:
  name: Autonomous UAV
  text: Flight algorithms, from scratch
  tagline: >
    Multirotor, fixed-wing and VTOL flight models with the physics written
    out in full — plus 40 runnable simulations and a gym for teaching a
    drone to fly itself.
  actions:
    - theme: brand
      text: First flight
      link: /guide/getting-started
    - theme: alt
      text: Algorithm atlas
      link: /simulations/
    - theme: alt
      text: GitHub
      link: https://github.com/guilyx/autonomous-uav-guide

features:
  - icon: ✈️
    title: Three airframes, one convention
    details: >
      A 6DOF quadrotor with motor dynamics, a full Beard & McLain fixed-wing
      with every stability derivative live, and a tilt-rotor VTOL that
      actually transitions. All in one frame convention, so they compose.
    link: /vehicles/
    linkText: Flight models

  - icon: 🎯
    title: Trim solvers and autopilots
    details: >
      Ask for 35 m/s and a 2 m/s climb and get back the exact elevator and
      throttle that make it an equilibrium. The autopilot then derives its
      own gains from the airframe — no retuning per aircraft.
    link: /vehicles/trim
    linkText: Trim and equilibrium

  - icon: 🧠
    title: A gym that teaches flight
    details: >
      Six reinforcement-learning tasks on the same physics, from hovering to
      flying a fixed-wing route without stalling. The trainer is pure NumPy,
      so a bare install can learn to fly.
    link: /learning/
    linkText: Reinforcement learning

  - icon: 🗺️
    title: Planning and estimation
    details: >
      A*, RRT*, PRM, potential fields and coverage in 3D. EKF, UKF,
      particle and complementary filters. Min-snap, Frenet and quintic
      trajectories. Each with a simulation you can run.
    link: /simulations/
    linkText: Browse the atlas

  - icon: 🐝
    title: Swarms
    details: >
      Reynolds flocking, consensus formation, virtual structures,
      leader-follower and Voronoi coverage — multi-agent behaviour on the
      same vehicle models.
    link: /simulations/swarm/
    linkText: Swarm algorithms

  - icon: 🔬
    title: Reproducible, not decorative
    details: >
      Every parameter in a model is read by that model. Every claim in these
      pages has a test behind it. Where a number is representative rather
      than measured, the docs say so.
    link: /guide/conventions
    linkText: Conventions
---

<div style="max-width: 1152px; margin: 0 auto; padding: 0 24px 64px;">

## See it fly

<video class="uav-promo" controls playsinline preload="none" poster="/media/promo-poster.png" src="/media/promo.mp4"></video>

Fifty seconds, and every frame of it is simulated by
[`scripts/make_promo.py`](https://github.com/guilyx/autonomous-uav-guide/blob/main/scripts/make_promo.py)
at render time — the same models this library ships, integrated live. No
stock footage, and nothing that can drift out of sync with the code.

<StatBand :items="[
  { value: '40+', label: 'runnable simulations' },
  { value: '3', label: 'airframe families' },
  { value: '6', label: 'RL environments' },
  { value: '430+', label: 'tests' },
]" />

## Sixty seconds to first flight

```bash
pip install uav-sim

uav-sim doctor            # verify the install, run the physics self-checks
uav-sim list              # browse 40+ simulations
uav-sim run pid_hover     # render one to a GIF
uav-sim train hover       # teach a quadrotor to hold position
```

Or drive it from Python:

```python
from uav_sim.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset
from uav_sim.control.fixed_wing_autopilot import FixedWingAutopilot, AutopilotCommand

aircraft = create_fixed_wing(FixedWingPreset.SKYWALKER_X8)
aircraft.reset_trimmed(altitude=120.0)          # solve for equilibrium flight

pilot = FixedWingAutopilot(aircraft.fw_params)  # gains derived from the airframe
command = AutopilotCommand(altitude=160.0, airspeed=20.0, course=1.0)

for _ in range(12_000):
    aircraft.step(pilot.compute(aircraft.state, command, 0.01), 0.01)

print(aircraft.state[2])    # 160.0
```

## What "from scratch" means here

No solver is wrapped. No dynamics library is imported. The Newton-Euler
equations, the aerodynamic coefficient build-up, the Kalman recursions and
the sampling-based planners are all written out, in NumPy, next to the
citation they came from.

That is the point: the code is meant to be *read*. If a line needs a
comment to explain what it does, something is named wrong; if it needs one
to explain **why**, it has one.

::: tip A worked example
The [fixed-wing model](/vehicles/fixed-wing) is the best illustration. It
carries thirty-odd stability derivatives, and every one of them changes the
aircraft's behaviour — there is a test that fails if you zero any of them.
:::

## Where to go next

<CardGrid :items="[
  {
    title: 'First flight',
    meta: 'guide',
    link: '/guide/getting-started',
    body: 'Install, fly a quadrotor, render a simulation, and read the state vector.'
  },
  {
    title: 'Frames and conventions',
    meta: 'guide',
    link: '/guide/conventions',
    body: 'ENU world, FLU body, and why positive pitch means nose-down here. Read this before writing a controller.'
  },
  {
    title: 'Flight models',
    meta: 'vehicles',
    link: '/vehicles/',
    body: 'Quadrotor, fixed-wing and VTOL: what each models, and what each deliberately does not.'
  },
  {
    title: 'Training a policy',
    meta: 'learning',
    link: '/learning/training',
    body: 'How the trainer works, why the policy is linear, and the four choices that decide whether it learns at all.'
  },
  {
    title: 'Algorithm atlas',
    meta: 'reference',
    link: '/simulations/',
    body: 'Every simulation, grouped by domain, each with a preview and its source.'
  },
  {
    title: 'CLI reference',
    meta: 'guide',
    link: '/guide/cli',
    body: 'list, run, train, play, trim and doctor — the whole surface in one page.'
  }
]" />

</div>
