# CLI reference

```bash
flybots --help
flybots --version
```

Built on argparse, so the package stays dependency-free. Colour is disabled
automatically when output is not a terminal, when `NO_COLOR` is set, or
when `TERM=dumb`.

The command used to be `uav-sim`. That name still works — it warns on
stderr and forwards to `flybots` — but it will be removed in a later
release.

## `list` — browse the catalogue

```bash
flybots list
flybots list --category planning
```

Groups every simulation by domain, showing whether a preview GIF has been
rendered and the first line of its docstring. Simulations are discovered by
walking the package, so a new one appears here as soon as it has a `run.py`.

## `run` — render a simulation

```bash
flybots run pid_hover                    # by name
flybots run path_planning/astar_3d       # by slug
flybots run 'swarm/*' --all              # by glob
flybots run ekf --traceback              # full traceback on failure
```

Accepts a bare name, a `category/name` slug, or a glob. A pattern matching
several simulations lists them and stops unless `--all` is given.

Each simulation writes its GIF and a JSON log next to its `run.py`.

## `info` — details for one simulation

```bash
flybots info ekf
```

Prints the summary, module path, the equivalent `python -m` command, the
GIF location, and the simulation's README.

## `envs` — list RL environments

```bash
flybots envs
```

```text
id           vehicle     level   obs     act    description
hover        quadrotor   easy    obs 18  act 4  Hold a point in space...
waypoint     quadrotor   medium  obs 18  act 4  Fly to a random waypoint...
trajectory   quadrotor   hard    obs 21  act 4  Track a moving Lissajous...
landing      quadrotor   medium  obs 18  act 4  Descend and touch down...
fw-cruise    fixed-wing  medium  obs 16  act 4  Hold altitude, airspeed...
fw-waypoint  fixed-wing  hard    obs 16  act 4  Fly a fixed-wing route...
```

## `train` — learn a flight policy

```bash
flybots train hover
flybots train fw-cruise --iterations 200 --directions 16
flybots train trajectory --hidden 64 64
flybots train hover --optimizer cem --population 40
```

| Flag | Default | Meaning |
|---|---|---|
| `-i, --iterations` | 120 | Optimiser iterations |
| `-d, --directions` | 16 | ARS perturbation directions per iteration |
| `--optimizer` | `ars` | `ars` or `cem` |
| `-p, --population` | 40 | CEM candidates per generation |
| `-e, --episodes` | 8 | Episodes averaged per evaluation |
| `--hidden` | — | Hidden layer sizes; omit for a linear policy |
| `--seed` | 0 | Random seed |
| `--eval-episodes` | 20 | Held-out evaluation episodes |
| `-o, --output` | `policies/<env>.npz` | Where to save |

Prints a progress bar with the held-out return, then a full evaluation
including the share of episodes ending in each termination reason.

## `play` — roll out a policy

```bash
flybots play hover --policy policies/hover.npz
flybots play hover --policy policies/hover.npz --episodes 5 --gif hover.gif
flybots play landing                       # untrained zero policy
```

Reports per-episode return, length and termination reason. With `--gif`,
renders the flown trajectories against the goal.

## `trim` — fixed-wing trim table

```bash
flybots trim aerosonde
flybots trim skywalker_x8 --climb 2.0
```

```text
skywalker_x8 — trim envelope
────────────────────────────
  mass          3.36 kg
  wing area     0.750 m^2
  aspect ratio  5.88
  wing loading  44.0 N/m^2
  stall speed   8.4 m/s
  cruise speed  18.0 m/s

Va m/s  alpha deg  elev deg   throttle
   8.8    14.14       -25.62     0.448
  12.5     5.83        -7.74     0.449
  17.9     1.89         0.73     0.561
  25.2     0.05         4.69     0.799
```

Sweeps the speed envelope and solves for trim at each point. Speeds with no
solution are marked `unreachable` rather than silently omitted. See
[Trim and equilibrium](/vehicles/trim).

## `doctor` — check the install

```bash
flybots doctor
```

Reports the interpreter, dependency versions (marking optional ones as
such), the catalogue size, and then **flies each airframe** to confirm the
physics behaves:

```text
Physics self-check
──────────────────
  ok        fixed-wing holds trim    0.000 m drift
  ok        quadrotor hovers         0.000 m drift
  ok        VTOL hovers              0.000 m drift
```

Exits non-zero if a required dependency is missing or a self-check fails,
so it is usable in CI.

## Running simulations as modules

The CLI is a convenience. Every simulation is also a module:

```bash
python -m flybots.simulations.path_tracking.pid_hover
python -m flybots.simulations.estimation.ekf
```
