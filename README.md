<div align="center">

# flybots

**Flight algorithms, from scratch.**

Multirotor, fixed-wing and VTOL flight models with the physics written out
in full — plus 47 runnable simulations and a gym for teaching a drone to
fly itself.

[**Documentation**](https://guilyx.github.io/flybots/) ·
[Getting started](https://guilyx.github.io/flybots/guide/getting-started) ·
[Flight models](https://guilyx.github.io/flybots/vehicles/) ·
[Reinforcement learning](https://guilyx.github.io/flybots/learning/) ·
[Algorithm atlas](https://guilyx.github.io/flybots/simulations/)

[![CI](https://github.com/guilyx/flybots/actions/workflows/ci.yml/badge.svg)](https://github.com/guilyx/flybots/actions/workflows/ci.yml)
[![Docs](https://github.com/guilyx/flybots/actions/workflows/pages.yml/badge.svg)](https://guilyx.github.io/flybots/)
[![PyPI](https://img.shields.io/pypi/v/flybots.svg?logo=pypi&logoColor=white)](https://pypi.org/project/flybots/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/guilyx/flybots/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

</div>

<a href="https://guilyx.github.io/flybots/#see-it-fly">
  <img src="https://raw.githubusercontent.com/guilyx/flybots/main/docs/public/media/promo.gif" alt="flybots — quadrotor, fixed-wing, VTOL, planning, trajectory generation, estimation, mapping, swarms and reinforcement learning" width="820"/>
</a>

<sub>Quadrotor, fixed-wing, VTOL, planning, trajectory generation, estimation, mapping, swarms and reinforcement learning, closing on all 47 simulations. Every frame is simulated at render time by <a href="https://github.com/guilyx/flybots/blob/main/scripts/make_promo.py"><code>scripts/make_promo.py</code></a> — no stock footage, and nothing that can drift out of sync with the code. Above is a sample of each scene; the full seventy-eight seconds, at full resolution, is <a href="https://guilyx.github.io/flybots/#see-it-fly">on the docs site</a>.</sub>

---

## Install

```bash
pip install flybots
```

```bash
flybots doctor            # verify the install, run the physics self-checks
flybots list              # browse 47 simulations
flybots run pid_hover     # render one to a GIF
flybots train hover       # teach a quadrotor to hold position
```

## Sixty seconds in

A fixed wing, trimmed and flown to a new altitude and heading:

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

## What is here

| | |
|---|---|
| **Vehicles** | 6DOF quadrotor with motor dynamics · full Beard & McLain fixed-wing · tilt-rotor VTOL that transitions |
| **Control** | Cascaded PID · LQR · MPC · pure pursuit · geometric SO(3) · fixed-wing autopilot · VTOL mode scheduler |
| **Planning** | A* · RRT* · PRM · potential field · coverage · min-snap · Frenet · quintic |
| **Estimation** | EKF · UKF · particle filter · complementary filter · EKF-SLAM |
| **Perception** | Occupancy mapping · obstacle detection · visual servoing · gimbal tracking |
| **Swarm** | Reynolds flocking · consensus · virtual structure · leader-follower · Voronoi coverage |
| **Comms** | Algebraic connectivity maintenance · relay coverage · path-loss and Gaussian link models |
| **Safety** | Control barrier functions · CBF-QP safety filter · geofence, separation and speed barriers |
| **Learning** | 6 RL environments · pure-NumPy trainer (ARS, CEM) · optional Gymnasium integration |

Nothing here wraps a solver. The Newton-Euler equations, the aerodynamic
coefficient build-up, the Kalman recursions and the sampling-based planners
are written out in NumPy next to the citation they came from.

## Flight models

Three airframes, one frame convention, so they compose.

```python
from uav_sim.vehicles.fixed_wing import create_fixed_wing, FixedWingPreset

aircraft = create_fixed_wing(FixedWingPreset.AEROSONDE)
controls = aircraft.reset_trimmed(airspeed=35.0, altitude=200.0)

for _ in range(6000):
    aircraft.step(controls, 0.005)

aircraft.state[2]     # 200.0 — thirty seconds, open loop, no drift
```

That is the acceptance test for the whole aerodynamic model: if any force
or moment is inconsistent, trim is not an equilibrium and the aircraft
wanders. Every stability derivative is live, and there is a test that fails
if you zero any of them.

- [Fixed-wing](https://guilyx.github.io/flybots/vehicles/fixed-wing) — coefficient build-up, stall, trim solver
- [VTOL tilt-rotor](https://guilyx.github.io/flybots/vehicles/vtol) — hover → cruise → hover with altitude held
- [Quadrotor](https://guilyx.github.io/flybots/vehicles/quadrotor) — mixer and per-motor dynamics

## Teach one to fly

```bash
flybots envs                # 6 tasks: hover, waypoint, trajectory, landing, 2 fixed-wing
flybots train hover         # pure NumPy — no deep-learning stack
flybots play hover --policy policies/hover.npz --gif hover.gif
```

```python
from uav_sim.gym import make, train, evaluate

result = train("hover", iterations=120, seed=0)
print(evaluate("hover", result.policy, episodes=25))
```

Gymnasium's API without the Gymnasium dependency. Install
`flybots[gym]` and the environments register as `uav_sim/Hover-v0` for use
with any standard RL library.

The interesting part is not the algorithm — it is that
[four setup choices](https://guilyx.github.io/flybots/learning/design-notes)
decide whether these tasks are learnable at all. Each is documented with
the measurement that motivated it.

## Simulations

Forty-odd runnable demos, each with a three-panel animation, an academic
reference and a JSON log:

```bash
flybots list
flybots info astar_3d
flybots run astar_3d
```

Browse them all in the
[algorithm atlas](https://guilyx.github.io/flybots/simulations/).

## Conventions

Worth ten minutes before you write a controller:

| | Frame | Axes |
|---|---|---|
| World | **ENU** | `x` east, `y` north, `z` up |
| Body | **FLU** | `x` forward, `y` left, `z` up |

A consequence of Forward-Left-Up is that **positive pitch is nose-down**,
and because the world is ENU, **banking right decreases the heading**.
Aerodynamics texts use Forward-Right-Down; the library converts at the
boundary rather than rewriting the equations. Full details in
[Frames and conventions](https://guilyx.github.io/flybots/guide/conventions).

## Development

```bash
# The simulation GIFs live in Git LFS and are ~150 MB. Skipping them clones
# in a few seconds and leaves small pointer files in their place, which is
# what you want unless you are regenerating the GIFs themselves. Drop the
# prefix to fetch them, or run `git lfs pull` later.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/guilyx/flybots.git
cd flybots
uv sync --all-groups

uv run flybots doctor
uv run pytest

pre-commit install && pre-commit install --hook-type commit-msg
```

Contributions welcome — see [CONTRIBUTING.md](https://github.com/guilyx/flybots/blob/main/CONTRIBUTING.md) for the bar
a new algorithm has to clear, and
[CHANGELOG.md](https://github.com/guilyx/flybots/blob/main/CHANGELOG.md) for what has changed.

## Safety

These models are simplified, the controllers are not certified, and nothing
here has been validated against a real airframe. Do not fly hardware on
control code taken from this repository without independent verification.
See [SECURITY.md](https://github.com/guilyx/flybots/blob/main/SECURITY.md).

## License

MIT — see [LICENSE](https://github.com/guilyx/flybots/blob/main/LICENSE).
