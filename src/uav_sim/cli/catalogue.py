# Erwin Lejeune - 2026-02-18
"""Discovery of the runnable simulations.

Simulations live at ``uav_sim/simulations/<category>/<name>/run.py``. Rather
than maintaining a hand-written registry that drifts out of date, this
module walks the package and reads each simulation's metadata from its own
docstring and README.
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["SimulationEntry", "discover", "resolve", "categories", "SIMULATION_ROOT"]

SIMULATION_ROOT = Path(__file__).resolve().parent.parent / "simulations"


@dataclass(frozen=True)
class SimulationEntry:
    """One runnable simulation."""

    slug: str
    """``category/name``, e.g. ``path_planning/astar_3d``."""
    category: str
    name: str
    directory: Path

    @property
    def module(self) -> str:
        return f"uav_sim.simulations.{self.category}.{self.name}.run"

    @property
    def command(self) -> str:
        return f"python -m uav_sim.simulations.{self.category}.{self.name}"

    @property
    def gif(self) -> Path | None:
        candidate = self.directory / f"{self.name}.gif"
        return candidate if candidate.exists() else None

    @property
    def readme(self) -> Path | None:
        candidate = self.directory / "README.md"
        return candidate if candidate.exists() else None

    @property
    def summary(self) -> str:
        """First line of the run script's docstring.

        Read from the source text rather than by importing, so listing the
        catalogue stays fast and cannot be broken by an import error in one
        simulation.
        """
        run_file = self.directory / "run.py"
        if not run_file.exists():
            return ""
        try:
            text = run_file.read_text(encoding="utf-8")
        except OSError:
            return ""
        marker = '"""'
        start = text.find(marker)
        if start == -1:
            return ""
        body = text[start + len(marker) :]
        end = body.find(marker)
        docstring = body[:end] if end != -1 else body
        for line in docstring.strip().splitlines():
            if line.strip():
                return line.strip()
        return ""

    def load(self):
        """Import the simulation module and return it."""
        return importlib.import_module(self.module)


@lru_cache(maxsize=1)
def discover() -> tuple[SimulationEntry, ...]:
    """Every simulation found on disk, sorted by slug."""
    entries = []
    for run_file in sorted(SIMULATION_ROOT.rglob("run.py")):
        directory = run_file.parent
        relative = directory.relative_to(SIMULATION_ROOT)
        if len(relative.parts) != 2:
            continue
        category, name = relative.parts
        entries.append(
            SimulationEntry(
                slug=f"{category}/{name}",
                category=category,
                name=name,
                directory=directory,
            )
        )
    return tuple(entries)


def categories() -> dict[str, list[SimulationEntry]]:
    """Simulations grouped by category, preserving sort order."""
    grouped: dict[str, list[SimulationEntry]] = {}
    for entry in discover():
        grouped.setdefault(entry.category, []).append(entry)
    return grouped


def resolve(selector: str) -> list[SimulationEntry]:
    """Resolve a slug, a bare name, or a glob pattern to simulations.

    Accepts ``path_planning/astar_3d``, ``astar_3d`` and ``path_planning/*``,
    because remembering the category should not be a prerequisite for
    running something.
    """
    entries = discover()
    exact = [e for e in entries if e.slug == selector]
    if exact:
        return exact

    by_name = [e for e in entries if e.name == selector]
    if by_name:
        return by_name

    pattern = [
        e
        for e in entries
        if fnmatch.fnmatch(e.slug, selector) or fnmatch.fnmatch(e.name, selector)
    ]
    if pattern:
        return pattern

    suggestions = _closest(selector, [e.slug for e in entries])
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise KeyError(f"No simulation matches {selector!r}.{hint}")


def _closest(target: str, options: list[str], limit: int = 3) -> list[str]:
    """Cheap fuzzy suggestions based on shared character overlap."""
    import difflib

    return difflib.get_close_matches(target, options, n=limit, cutoff=0.4)
