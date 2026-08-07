# Contributing

Thanks for considering a contribution. This project is a reference
implementation of autonomous UAV algorithms — the goal is that every
algorithm is *correct*, *readable*, and *runnable*, in that order.

## Quick start

```bash
git clone https://github.com/guilyx/autonomous-uav-guide.git
cd autonomous-uav-guide
uv sync --all-groups
uv run uav-sim doctor      # verify the install and physics self-checks
uv run pytest
```

Set up the hooks once so formatting and commit messages are checked
locally rather than in CI:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

## The bar for a new algorithm

A new algorithm is not finished when it runs. It is finished when someone
who has never seen it can understand what it does and trust the result.
Concretely, a pull request adding one needs:

1. **An implementation** in the matching `src/uav_sim/<area>/` package,
   written from scratch — no wrapping of an external solver.
2. **A citation** in the module docstring: author, title, venue, year. If
   the parameters come from a published set, say which one.
3. **A runnable simulation** at
   `src/uav_sim/simulations/<area>/<name>/run.py` exposing `main()`, so
   `uav-sim run <name>` works.
4. **Tests** that pin the behaviour, not just the shape of the output.
5. **A docs page** under `docs/` and a sidebar entry in
   `docs/.vitepress/config.ts`.

### What "tests that pin the behaviour" means

The most valuable test is the one that would have caught a real bug. For
a dynamics model, that means asserting on physics:

```python
def test_trimmed_flight_holds_altitude_open_loop():
    """Trim is a real equilibrium, not just a plausible-looking state."""
    aircraft = create_fixed_wing(FixedWingPreset.AEROSONDE)
    controls = aircraft.reset_trimmed(altitude=300.0)
    for _ in range(6000):
        aircraft.step(controls, 0.005)
    assert aircraft.state[2] == pytest.approx(300.0, abs=1.0)
```

Asserting that `step()` returns a 12-element array would pass against a
model that integrates altitude the wrong way. Asserting that the aircraft
still flies level after thirty seconds would not.

If a parameter appears in a dataclass, something should fail when it
changes. Unused parameters are how a model ends up looking more
sophisticated than it is.

## Conventions

These are load-bearing. Getting them wrong produces models that look
right and fly backwards.

| Quantity | Convention |
|---|---|
| World frame | **ENU** — `z` is altitude, increasing upward |
| Body frame | **FLU** — `x` forward, `y` left, `z` up |
| Euler angles | ZYX `(roll, pitch, yaw)`; `rotation_matrix` maps body → world |
| Pitch sign | **Positive `theta` is nose-down** (a consequence of FLU) |
| Yaw sign | Increases counter-clockwise; banking right *decreases* it |
| Angles | Radians everywhere in the API; degrees only in display strings |
| Units | SI throughout — metres, seconds, kilograms, newtons |

Aerodynamics texts use Forward-Right-Down instead. When porting textbook
equations, convert at the boundary with
`uav_sim.frames.transforms.flu_to_frd` rather than rewriting the
equations — see `vehicles/fixed_wing/aerodynamics.py` for the pattern.

## Style

- Ruff handles formatting and linting: `pre-commit run --all-files`.
- Line length is 99.
- Public functions get type hints and a docstring with units.
- Comments explain *why*, not *what*. If a line needs a comment to say
  what it does, rename something instead.

## Commits

Conventional Commits, enforced by commitizen on `commit-msg`:

```
feat(planning): add informed RRT*
fix(vehicles): correct fixed-wing gravity sign in body frame
docs(site): add trim solver walkthrough
test(gym): cover landing termination cases
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`,
`build`, `ci`, `chore`.

## Pull requests

- Every PR must pass `pre-commit run --all-files` and `uv run pytest`.
- Describe *what changed and why*. If you fixed a bug, say what the
  incorrect behaviour was and how you know it is fixed.
- Regenerating GIFs is optional — say so if you did, since they are large
  and stored via Git LFS.

## Reporting a bug

Physics bugs are the interesting ones. A good report includes the state
you started from, the input you applied, what happened, and what should
have happened. A ten-line script beats a paragraph:

```python
aircraft = FixedWing()
aircraft.reset_trimmed(airspeed=35.0, altitude=200.0)
# expected: holds 200 m. actual: climbs away.
```

## Questions

Open a [discussion](https://github.com/guilyx/autonomous-uav-guide/discussions)
or an issue. Questions about *why* an algorithm is implemented a
particular way are welcome — if the answer is not obvious from the code,
that is a documentation bug.
