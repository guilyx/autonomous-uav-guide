# Erwin Lejeune - 2026-02-19
"""Regenerate all simulation GIFs by running each simulation's main()."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parent.parent / "src" / "uav_sim" / "simulations"

SIMS = sorted(str(p.parent.relative_to(SIM_ROOT)) for p in SIM_ROOT.rglob("run.py"))


def _module_name(rel_path: str) -> str:
    return "uav_sim.simulations." + rel_path.replace("/", ".") + ".run"


def main() -> None:
    failed: list[str] = []
    for i, sim in enumerate(SIMS, 1):
        mod_name = _module_name(sim)
        print(f"\n[{i}/{len(SIMS)}] {sim} ({mod_name})", flush=True)
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
            print("  -> OK", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(sim)
            print("  -> FAILED", flush=True)

    print(f"\n{'='*60}")
    print(f"Done: {len(SIMS) - len(failed)}/{len(SIMS)} succeeded")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
