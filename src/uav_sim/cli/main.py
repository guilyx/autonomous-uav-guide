# Erwin Lejeune - 2026-02-18
"""``uav-sim`` command-line interface.

    uav-sim list                     browse the simulation catalogue
    uav-sim run pid_hover            run a simulation and render its GIF
    uav-sim info ekf                 show references and usage for one
    uav-sim envs                     list the reinforcement-learning tasks
    uav-sim train hover              train a flight policy from scratch
    uav-sim play hover --policy p    roll a trained policy out
    uav-sim trim aerosonde           print a fixed-wing trim table
    uav-sim doctor                   check the local install

Built on argparse so the package stays dependency-free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uav_sim import __version__
from uav_sim.cli import catalogue
from uav_sim.cli.console import heading, style, table

__all__ = ["main", "build_parser"]

# (module name, required). Checked by `doctor` -- fixed at import time, never
# built from a command-line argument or any other external input.
_DEPENDENCY_CHECKS = (
    ("numpy", True),
    ("scipy", True),
    ("matplotlib", True),
    ("gymnasium", False),
    ("imageio_ffmpeg", False),
)

_BANNER = r"""
   __  _____ _   __  ______
  / / / /   | | / / / ___/ /__ _
 / /_/ / /| | |/ /  \__ \  _  ' \
 \____/_/ |_|___/  /____/_/_/_/_/   autonomous uav toolkit
"""


def _print_banner() -> None:
    print(style(_BANNER, "sky"))
    print(f"  {style('uav-sim', 'bold')} {style(__version__, 'dim')}")


# ── list ──────────────────────────────────────────────────────────────────


def _cmd_list(args: argparse.Namespace) -> int:
    grouped = catalogue.categories()
    if args.category:
        grouped = {k: v for k, v in grouped.items() if args.category in k}
        if not grouped:
            print(style(f"No category matching {args.category!r}.", "red"), file=sys.stderr)
            return 1

    total = 0
    for category, entries in grouped.items():
        print(heading(category.replace("_", " ").title()))
        rows = []
        for entry in entries:
            marker = style("gif", "green") if entry.gif else style("  -", "dim")
            summary = entry.summary
            if len(summary) > 62:
                summary = summary[:59] + "..."
            rows.append((style(entry.name, "cyan"), marker, style(summary, "slate")))
            total += 1
        print(table(rows))

    print(
        f"\n{style(str(total), 'bold')} simulations. Run one with "
        f"{style('uav-sim run <name>', 'amber')}."
    )
    return 0


# ── run ───────────────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        entries = catalogue.resolve(args.simulation)
    except KeyError as error:
        print(style(str(error), "red"), file=sys.stderr)
        return 1

    if len(entries) > 1 and not args.all:
        print(style(f"{args.simulation!r} matches {len(entries)} simulations:", "amber"))
        for entry in entries:
            print(f"  {entry.slug}")
        print(f"\nRe-run with {style('--all', 'amber')} to run every match.")
        return 1

    failures = 0
    for entry in entries:
        print(f"\n{style('running', 'sky')} {style(entry.slug, 'bold')}")
        try:
            module = entry.load()
            module.main()
        except Exception as error:  # noqa: BLE001 - report and keep going
            failures += 1
            print(style(f"  failed: {type(error).__name__}: {error}", "red"), file=sys.stderr)
            if args.traceback:
                import traceback

                traceback.print_exc()
            continue
        if entry.gif:
            print(
                style(f"  wrote {entry.gif.relative_to(Path.cwd())}", "green")
                if entry.gif.is_relative_to(Path.cwd())
                else style(f"  wrote {entry.gif}", "green")
            )
    return 1 if failures else 0


# ── info ──────────────────────────────────────────────────────────────────


def _cmd_info(args: argparse.Namespace) -> int:
    try:
        entries = catalogue.resolve(args.simulation)
    except KeyError as error:
        print(style(str(error), "red"), file=sys.stderr)
        return 1

    for entry in entries:
        print(heading(entry.slug))
        print(f"  {style('summary', 'slate')}  {entry.summary}")
        print(f"  {style('module ', 'slate')}  {entry.module}")
        print(f"  {style('command', 'slate')}  {style(entry.command, 'amber')}")
        print(f"  {style('gif    ', 'slate')}  {entry.gif or style('not rendered yet', 'dim')}")
        if entry.readme:
            print(f"\n{entry.readme.read_text(encoding='utf-8').strip()}")
    return 0


# ── envs ──────────────────────────────────────────────────────────────────


def _cmd_envs(args: argparse.Namespace) -> int:
    from uav_sim.gym import list_envs, make

    print(heading("Reinforcement-learning environments"))
    rows = []
    for spec in list_envs():
        env = make(spec.env_id)
        rows.append(
            (
                style(spec.env_id, "cyan"),
                spec.vehicle,
                _difficulty(spec.difficulty),
                f"obs {env.observation_space.shape[0]}",
                f"act {env.action_space.shape[0]}",
                style(spec.description, "slate"),
            )
        )
        env.close()
    print(table(rows, headers=("id", "vehicle", "level", "obs", "act", "description")))
    print(f"\nTrain one with {style('uav-sim train <id>', 'amber')}.")
    return 0


def _difficulty(level: str) -> str:
    return style(level, {"easy": "green", "medium": "amber", "hard": "red"}.get(level, "slate"))


# ── train ─────────────────────────────────────────────────────────────────


def _cmd_train(args: argparse.Namespace) -> int:
    from uav_sim.gym import evaluate, train
    from uav_sim.gym.train import TrainConfig

    config = TrainConfig(
        optimizer=args.optimizer,
        iterations=args.iterations,
        directions=args.directions,
        population=args.population,
        episodes_per_candidate=args.episodes,
        hidden_sizes=tuple(args.hidden) if args.hidden else (),
        seed=args.seed,
    )

    print(heading(f"Training '{args.env}'"))
    method = {
        "ars": "augmented random search",
        "cem": "cross-entropy method",
    }.get(config.optimizer.lower(), config.optimizer)
    print(f"  {style('method', 'slate')}      {method}")
    print(
        f"  {style('policy', 'slate')}      "
        f"{'linear' if not config.hidden_sizes else f'MLP {config.hidden_sizes}'}"
    )
    print(f"  {style('budget', 'slate')}      {config.iterations} iterations\n")

    width = 34

    latest = {"return": float("nan")}

    def progress(row: dict[str, float]) -> None:
        iteration = int(row["iteration"])
        import math

        if not math.isnan(row["holdout_return"]):
            latest["return"] = row["holdout_return"]
        filled = int(width * (iteration + 1) / config.iterations)
        bar = style("█" * filled, "sky") + style("░" * (width - filled), "dim")
        print(
            f"\r  {bar} iter {iteration + 1:>4}/{config.iterations}  "
            f"return {latest['return']:>9.1f}",
            end="",
            flush=True,
        )

    try:
        result = train(args.env, config=config, progress=progress)
    except KeyError as error:
        print(style(f"\n{error}", "red"), file=sys.stderr)
        return 1
    print()

    print(
        f"\n  {style('trained in', 'slate')}  {result.elapsed_seconds:.1f}s "
        f"({result.total_episodes} episodes)"
    )
    print(f"  {style('best return', 'slate')} {style(f'{result.best_return:.1f}', 'bold')}")

    summary = evaluate(args.env, result.policy, episodes=args.eval_episodes, seed=12345)
    print(heading("Held-out evaluation"))
    for key, value in summary.items():
        print(f"  {style(key.ljust(20), 'slate')} {value:.2f}")

    destination = Path(args.output or f"policies/{args.env}.npz")
    result.policy.save(destination)
    print(f"\n  {style('saved', 'green')} {destination}")
    print(f"  Replay it with {style(f'uav-sim play {args.env} --policy {destination}', 'amber')}")
    return 0


# ── play ──────────────────────────────────────────────────────────────────


def _cmd_play(args: argparse.Namespace) -> int:
    import numpy as np

    from uav_sim.gym import make
    from uav_sim.gym.policy import MLPPolicy
    from uav_sim.gym.train import rollout

    env = make(args.env, seed=args.seed)
    if args.policy:
        policy = MLPPolicy.load(args.policy)
    else:
        policy = MLPPolicy(
            env.observation_space.shape[0], env.action_space.shape[0], hidden_sizes=()
        )
        print(style("No --policy given; flying the untrained zero policy.", "amber"))

    print(heading(f"Playing '{args.env}'"))
    returns, lengths = [], []
    trajectories = []
    for episode in range(args.episodes):
        episode_return, steps, info = rollout(env, policy, seed=args.seed + episode)
        returns.append(episode_return)
        lengths.append(steps)
        trajectories.append(env.trajectory)
        reason = info.get("termination_reason", "time_limit")
        print(
            f"  episode {episode + 1:>2}  return {episode_return:>9.1f}  "
            f"steps {steps:>4}  {style(reason, 'slate')}"
        )

    print(
        f"\n  {style('mean return', 'slate')} {np.mean(returns):.1f}"
        f"   {style('mean length', 'slate')} {np.mean(lengths):.0f}"
    )

    if args.gif:
        from uav_sim.gym.render import render_episode

        path = render_episode(env, trajectories, args.gif, title=f"{args.env} policy")
        print(f"  {style('wrote', 'green')} {path}")
    env.close()
    return 0


# ── trim ──────────────────────────────────────────────────────────────────


def _cmd_trim(args: argparse.Namespace) -> int:
    import numpy as np

    from uav_sim.vehicles.fixed_wing import (
        FixedWingPreset,
        TrimError,
        compute_trim,
        get_fixed_wing_params,
    )

    try:
        preset = FixedWingPreset(args.preset)
    except ValueError:
        available = ", ".join(p.value for p in FixedWingPreset if p is not FixedWingPreset.CUSTOM)
        print(
            style(f"Unknown airframe {args.preset!r}. Available: {available}", "red"),
            file=sys.stderr,
        )
        return 1

    params = get_fixed_wing_params(preset)
    print(heading(f"{preset.value} — trim envelope"))
    print(f"  {style('mass', 'slate')}          {params.mass:.2f} kg")
    print(f"  {style('wing area', 'slate')}     {params.wing_area:.3f} m^2")
    print(f"  {style('aspect ratio', 'slate')}  {params.aspect_ratio:.2f}")
    print(f"  {style('wing loading', 'slate')}  {params.wing_loading():.1f} N/m^2")
    print(f"  {style('stall speed', 'slate')}   {params.stall_airspeed:.1f} m/s")
    print(f"  {style('cruise speed', 'slate')}  {params.cruise_airspeed:.1f} m/s\n")

    speeds = np.linspace(params.stall_airspeed * 1.05, params.cruise_airspeed * 1.4, 10)
    rows = []
    for airspeed in speeds:
        try:
            trim = compute_trim(params, airspeed=float(airspeed), climb_rate=args.climb)
        except TrimError:
            rows.append((f"{airspeed:6.1f}", style("unreachable", "red"), "", ""))
            continue
        rows.append(
            (
                f"{airspeed:6.1f}",
                f"{trim.alpha_deg:7.2f}",
                f"{np.degrees(trim.elevator):9.2f}",
                f"{trim.throttle:8.3f}",
            )
        )
    print(table(rows, headers=("Va m/s", "alpha deg", "elev deg", "throttle")))
    return 0


# ── doctor ────────────────────────────────────────────────────────────────


def _cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    print(heading("Environment"))
    print(f"  {style('uav-sim', 'slate')}   {__version__}")
    print(f"  {style('python', 'slate')}    {platform.python_version()}")
    print(f"  {style('platform', 'slate')}  {platform.platform()}")

    print(heading("Dependencies"))
    ok = True
    for name, required in _DEPENDENCY_CHECKS:
        found = importlib.util.find_spec(name) is not None
        if found:
            try:
                version = importlib.import_module(name).__version__
            except Exception:  # noqa: BLE001
                # The package is installed, which is what `doctor` is checking.
                # Reporting its version is a nicety, not a diagnosis.
                version = "version unknown"
            mark, colour = "ok", "green"
        else:
            version = "not installed"
            mark, colour = ("missing", "red") if required else ("optional", "dim")
            ok &= not required
        print(f"  {style(mark.ljust(9), colour)} {name.ljust(16)} {style(version, 'slate')}")

    print(heading("Catalogue"))
    entries = catalogue.discover()
    rendered = sum(1 for e in entries if e.gif)
    print(f"  {len(entries)} simulations, {rendered} with rendered previews")

    from uav_sim.gym import list_envs

    print(f"  {len(list_envs())} reinforcement-learning environments")

    print(heading("Physics self-check"))
    ok &= _selfcheck()
    return 0 if ok else 1


def _selfcheck() -> bool:
    """Fly each airframe briefly and confirm it behaves."""
    import numpy as np

    checks = []

    try:
        from uav_sim.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing

        aircraft = create_fixed_wing(FixedWingPreset.AEROSONDE)
        controls = aircraft.reset_trimmed(altitude=200.0)
        for _ in range(2000):
            aircraft.step(controls, 0.005)
        drift = abs(aircraft.state[2] - 200.0)
        checks.append(("fixed-wing holds trim", drift < 1.0, f"{drift:.3f} m drift"))
    except Exception as error:  # noqa: BLE001
        checks.append(("fixed-wing holds trim", False, str(error)))

    try:
        from uav_sim.vehicles.multirotor import Quadrotor

        quad = Quadrotor()
        quad.reset(position=np.array([0.0, 0.0, 2.0]))
        hover = quad.hover_wrench()
        for motor in quad.motors:
            motor.reset(motor.thrust_to_omega(hover[0] / 4.0))
        for _ in range(500):
            quad.step(hover, 0.005)
        drift = abs(quad.state[2] - 2.0)
        checks.append(("quadrotor hovers", drift < 0.2, f"{drift:.3f} m drift"))
    except Exception as error:  # noqa: BLE001
        checks.append(("quadrotor hovers", False, str(error)))

    try:
        from uav_sim.vehicles.vtol import Tiltrotor

        vtol = Tiltrotor()
        state = np.zeros(12)
        state[2] = 20.0
        vtol.reset(state=state)
        weight = vtol.vtol_params.mass * vtol.vtol_params.gravity
        for _ in range(500):
            vtol.step(np.array([weight, 0, 0, 0, 0.0]), 0.005)
        drift = abs(vtol.state[2] - 20.0)
        checks.append(("VTOL hovers", drift < 0.2, f"{drift:.3f} m drift"))
    except Exception as error:  # noqa: BLE001
        checks.append(("VTOL hovers", False, str(error)))

    all_ok = True
    for label, passed, detail in checks:
        mark, colour = ("ok", "green") if passed else ("FAIL", "red")
        print(f"  {style(mark.ljust(9), colour)} {label.ljust(24)} {style(detail, 'slate')}")
        all_ok &= passed
    return all_ok


# ── parser ────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uav-sim",
        description="Autonomous UAV algorithms: simulations, flight models and RL tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uav-sim list --category planning\n"
            "  uav-sim run astar_3d\n"
            "  uav-sim train hover --generations 60\n"
            "  uav-sim trim skywalker_x8\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"uav-sim {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="list available simulations")
    p_list.add_argument("-c", "--category", help="filter by category substring")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser("run", help="run a simulation and render its GIF")
    p_run.add_argument("simulation", help="slug, name or glob pattern")
    p_run.add_argument("--all", action="store_true", help="run every match")
    p_run.add_argument("--traceback", action="store_true", help="show full tracebacks")
    p_run.set_defaults(func=_cmd_run)

    p_info = sub.add_parser("info", help="show details for a simulation")
    p_info.add_argument("simulation")
    p_info.set_defaults(func=_cmd_info)

    p_envs = sub.add_parser("envs", help="list reinforcement-learning environments")
    p_envs.set_defaults(func=_cmd_envs)

    p_train = sub.add_parser("train", help="train a flight policy")
    p_train.add_argument("env", help="environment id, see 'uav-sim envs'")
    p_train.add_argument("-i", "--iterations", type=int, default=120)
    p_train.add_argument(
        "-d",
        "--directions",
        type=int,
        default=16,
        help="ARS perturbation directions per iteration",
    )
    p_train.add_argument("--optimizer", choices=["ars", "cem"], default="ars")
    p_train.add_argument(
        "-p", "--population", type=int, default=40, help="CEM candidates per generation"
    )
    p_train.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=8,
        help="episodes averaged per evaluation (fewer is faster and worse)",
    )
    p_train.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=None,
        help="hidden layer sizes; omit for a linear policy",
    )
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--eval-episodes", type=int, default=20)
    p_train.add_argument("-o", "--output", help="where to save the policy")
    p_train.set_defaults(func=_cmd_train)

    p_play = sub.add_parser("play", help="roll out a policy")
    p_play.add_argument("env")
    p_play.add_argument("--policy", help="path to a saved .npz policy")
    p_play.add_argument("--episodes", type=int, default=5)
    p_play.add_argument("--seed", type=int, default=0)
    p_play.add_argument("--gif", help="render the episodes to this GIF")
    p_play.set_defaults(func=_cmd_play)

    p_trim = sub.add_parser("trim", help="print a fixed-wing trim table")
    p_trim.add_argument("preset", nargs="?", default="aerosonde")
    p_trim.add_argument("--climb", type=float, default=0.0, help="climb rate [m/s]")
    p_trim.set_defaults(func=_cmd_trim)

    p_doctor = sub.add_parser("doctor", help="check the local install")
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        _print_banner()
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print(style("\ninterrupted", "amber"), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
