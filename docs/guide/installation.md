# Installation

## Requirements

Python 3.12 or newer. The core install needs only NumPy, SciPy and
Matplotlib.

## From PyPI

```bash
pip install flybots
```

Optional extras:

```bash
pip install "flybots[gym]"     # Gymnasium integration for external RL libraries
pip install "flybots[video]"   # MP4 export via a bundled ffmpeg
pip install "flybots[gym,video]"
```

Neither extra is needed to train a policy — the built-in trainer is pure
NumPy. `gym` only matters if you want to drive the environments with
Stable-Baselines3, CleanRL or similar.

::: warning Renamed from `uav-sim`
The distribution and its command are now `flybots`. Installing `uav-sim`
still leaves a `uav-sim` command on your PATH — it prints a deprecation
notice on stderr and then runs `flybots`, and it will be removed in a later
release.

The *import* package has not been renamed yet: it is still `uav_sim`, so
`from uav_sim.vehicles.multirotor import Quadrotor` is unchanged, as are the
Gymnasium environment ids (`uav_sim/Hover-v0`). That rename is coming, and
will land with its own major-version bump and migration notes.
:::

## From source

The project uses [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/guilyx/autonomous-uav-guide.git
cd autonomous-uav-guide
uv sync --all-groups
```

`uv sync` reads `.python-version`, fetches the right interpreter, and
installs the package in editable mode with the dev tooling.

Plain pip works too:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gym,video]"
pip install pytest ruff pre-commit
```

::: tip Git LFS
The repository stores its preview GIFs in Git LFS. If you only want the
code, `GIT_LFS_SKIP_SMUDGE=1 git clone ...` skips roughly 100 MB of media.
:::

## Verify

```bash
flybots doctor
```

This prints the interpreter and dependency versions, counts the
simulation catalogue, and then flies each airframe briefly to confirm the
physics behaves:

```text
Physics self-check
──────────────────
  ok        fixed-wing holds trim    0.000 m drift
  ok        quadrotor hovers         0.000 m drift
  ok        VTOL hovers              0.000 m drift
```

A `FAIL` here means something is genuinely wrong with the install rather
than with your code — please
[open an issue](https://github.com/guilyx/autonomous-uav-guide/issues/new/choose)
with the full output.

## Run the tests

```bash
uv run pytest              # full suite
uv run pytest -q -x        # stop at the first failure
uv run pytest tests/test_fixed_wing_aero.py -v
```

The suite runs real simulations rather than mocks, so it takes a few
minutes. That is the point — the tests assert on flight behaviour, not on
array shapes.

## Development setup

```bash
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit run --all-files
```

Ruff handles formatting and linting; commitizen enforces
[Conventional Commits](https://www.conventionalcommits.org/) on the commit
message. See [CONTRIBUTING.md](https://github.com/guilyx/autonomous-uav-guide/blob/main/CONTRIBUTING.md).

## Building the docs

```bash
cd docs
npm ci
npm run dev      # http://localhost:5173
npm run build
```

## Headless environments

Every simulation sets the Matplotlib `Agg` backend explicitly, so rendering
works over SSH and in CI with no display. Nothing opens a window.
