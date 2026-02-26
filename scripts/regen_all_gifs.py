# Erwin Lejeune - 2026-02-19
"""Regenerate all simulation GIFs by running each simulation's main()."""

from __future__ import annotations

import fnmatch
import importlib
import json
import sys
import traceback
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parent.parent / "src" / "uav_sim" / "simulations"
CONFIG_PATH = Path(__file__).resolve().parent / "regen_all_gifs.json"
AVAILABLE_SIMS = sorted(str(p.parent.relative_to(SIM_ROOT)) for p in SIM_ROOT.rglob("run.py"))


def _module_name(rel_path: str) -> str:
    return "uav_sim.simulations." + rel_path.replace("/", ".") + ".run"


def _expand_selector(selector: str) -> list[str]:
    if selector in AVAILABLE_SIMS:
        return [selector]
    matches = [s for s in AVAILABLE_SIMS if fnmatch.fnmatch(s, selector)]
    if matches:
        return matches
    raise ValueError(
        f"Unknown simulation selector '{selector}'. "
        "Use an exact name (e.g. 'estimation/ekf') or pattern (e.g. 'perception/*')."
    )


def _load_sims_from_config(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. "
            'Create it with JSON shape: {"sims": ["estimation/ekf", ...]}.'
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    sims = data.get("sims")
    if not isinstance(sims, list) or not all(isinstance(s, str) for s in sims):
        raise ValueError(f"Invalid config {config_path}: 'sims' must be a list of strings.")
    if not sims:
        return []

    expanded: list[str] = []
    seen: set[str] = set()
    for selector in sims:
        for sim in _expand_selector(selector):
            if sim not in seen:
                expanded.append(sim)
                seen.add(sim)
    return expanded


def main() -> None:
    sims = _load_sims_from_config(CONFIG_PATH)
    if not sims:
        print(f"No simulations selected in {CONFIG_PATH}. Nothing to do.")
        return

    failed: list[str] = []
    for i, sim in enumerate(sims, 1):
        mod_name = _module_name(sim)
        print(f"\n[{i}/{len(sims)}] {sim} ({mod_name})", flush=True)
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
            print("  -> OK", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(sim)
            print("  -> FAILED", flush=True)

    print(f"\n{'='*60}")
    print(f"Done: {len(sims) - len(failed)}/{len(sims)} succeeded")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
