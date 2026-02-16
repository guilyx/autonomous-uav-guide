# Erwin Lejeune - 2026-02-16

# Contributing

Contributions are welcome! This guide covers the development workflow.

## Prerequisites

- [Python 3.12+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) (package manager)
- [pre-commit](https://pre-commit.com/)

## Setup

```bash
# Clone the repo
git clone https://github.com/guilyx/autonomous-quadrotor-guide.git
cd autonomous-quadrotor-guide

# Install dependencies (creates .venv automatically)
uv sync --all-groups

# Install git hooks
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Development workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```

2. Make your changes. Every commit must follow
   [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(control): add sliding mode controller
   fix(models): correct drag coefficient sign
   docs: update swarm algorithms reference
   test: add coverage for EKF estimation
   ```

3. Run checks locally:
   ```bash
   pre-commit run --all-files   # lint + format + tests
   uv run pytest --cov          # full test suite
   uv run mypy src/quadrotor_sim
   ```

4. Push and open a Pull Request against `main`.

## Project structure

- `src/quadrotor_sim/` -- importable library (models, control, planning, swarm, etc.)
- `simulations/` -- runnable demo scripts
- `tests/` -- pytest test suite
- `docs/` -- algorithm documentation and references

## Adding a new algorithm

1. Create the module under the appropriate sub-package (e.g. `src/quadrotor_sim/control/`).
2. Add tests in `tests/`.
3. If it warrants a demo, add a simulation script in `simulations/`.
4. Document it in the relevant `docs/*.md` file.
5. Update `docs/algorithms.md` with the new entry.

## Code style

- Formatted and linted by **ruff** (runs automatically via pre-commit).
- Type hints are encouraged; `mypy` runs in CI.
- Docstrings use Google style.
