# Erwin Lejeune - 2026-02-18
"""Terminal styling helpers.

Deliberately tiny and dependency-free. Colour is disabled automatically
when the output is not a TTY, when ``NO_COLOR`` is set (see
https://no-color.org), or when ``TERM=dumb`` — so piping ``flybots list``
into a file gives clean text.
"""

from __future__ import annotations

import os
import shutil
import sys

__all__ = ["style", "supports_colour", "heading", "table", "PALETTE"]

# Matches the documentation site's palette so the CLI and the docs read as
# one product rather than two.
PALETTE = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "sky": "\033[38;5;39m",
    "cyan": "\033[38;5;44m",
    "amber": "\033[38;5;214m",
    "green": "\033[38;5;42m",
    "red": "\033[38;5;203m",
    "violet": "\033[38;5;141m",
    "slate": "\033[38;5;245m",
}


def supports_colour(stream=None) -> bool:
    """True when it is safe to emit ANSI escapes."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def style(text: str, *names: str, stream=None) -> str:
    """Wrap ``text`` in the named styles, or return it unchanged without colour."""
    if not names or not supports_colour(stream):
        return text
    codes = "".join(PALETTE.get(name, "") for name in names)
    return f"{codes}{text}{PALETTE['reset']}"


def heading(text: str) -> str:
    """A section heading with a rule under it, sized to the terminal."""
    width = min(shutil.get_terminal_size((80, 24)).columns, 88)
    rule = "─" * min(len(text), width)
    return f"\n{style(text, 'bold', 'sky')}\n{style(rule, 'dim')}"


def table(rows: list[tuple[str, ...]], *, headers: tuple[str, ...] | None = None) -> str:
    """Render fixed-width columns, sized to the widest cell in each."""
    if not rows:
        return ""
    all_rows = ([headers] if headers else []) + rows
    widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(all_rows[0]))]

    lines = []
    if headers:
        lines.append(
            style(
                "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)),
                "bold",
                "slate",
            )
        )
    for row in rows:
        lines.append("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)
