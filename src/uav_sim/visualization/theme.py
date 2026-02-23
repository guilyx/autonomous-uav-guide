# Erwin Lejeune — 2026-02-15
"""LinkedStein-inspired dark theme for UAV simulation visualisations.

Palette derived from the LinkedStein project colour scheme: deep navy
backgrounds, slate-blue accents, and muted earthy tones for data traces.
Every visualisation module calls :func:`apply_theme` before creating
figures so the entire suite shares a consistent, non-glowy aesthetic.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# ── LinkedStein core palette ──────────────────────────────────────────────────
INK_BLACK = "#0d1b2a"
PRUSSIAN_BLUE = "#1b263b"
DUSK_BLUE = "#415a77"
LAVENDER_GREY = "#778da9"
ALABASTER = "#e0e1dd"

# ── Supplementary muted accents (data traces, markers) ───────────────────────
SAGE = "#a3b18a"
TERRA_COTTA = "#bc6c5c"
AMBER = "#dda15e"
SLATE_ROSE = "#b07d9e"
FADED_TEAL = "#6b9e9b"

# ── Semantic aliases ──────────────────────────────────────────────────────────
C_BG_FIGURE = INK_BLACK
C_BG_AXES = PRUSSIAN_BLUE
C_GRID = DUSK_BLUE
C_TEXT = ALABASTER
C_TICK = LAVENDER_GREY

C_TRAIL = LAVENDER_GREY
C_REFERENCE = ALABASTER
C_PATH = DUSK_BLUE
C_START = SAGE
C_GOAL = TERRA_COTTA
C_ERROR = TERRA_COTTA
C_SPEED = FADED_TEAL
C_ACCENT1 = TERRA_COTTA
C_ACCENT2 = SAGE
C_ACCENT3 = AMBER

C_BUILDING_FACE = DUSK_BLUE
C_BUILDING_EDGE = LAVENDER_GREY

C_ARM1 = LAVENDER_GREY
C_ARM2 = ALABASTER
C_MOTOR = DUSK_BLUE
C_HUB = ALABASTER

C_LIDAR = DUSK_BLUE
C_LIDAR_HIT = TERRA_COTTA
C_CAMERA = AMBER
C_LIDAR3D = LAVENDER_GREY

C_TRUE = ALABASTER
C_ESTIMATE = LAVENDER_GREY
C_SIGMA_BAND = LAVENDER_GREY

C_X = TERRA_COTTA
C_Y = SAGE
C_Z = LAVENDER_GREY
C_THRUST = LAVENDER_GREY
C_HOVER_REF = DUSK_BLUE

# ── Costmap ───────────────────────────────────────────────────────────────────
COSTMAP_CMAP = "YlOrRd"

# ── Property cycle for default plot() colours ─────────────────────────────────
PROP_CYCLE = cycler(
    color=[
        LAVENDER_GREY,
        TERRA_COTTA,
        SAGE,
        AMBER,
        FADED_TEAL,
        SLATE_ROSE,
        ALABASTER,
        DUSK_BLUE,
    ]
)

_THEME_APPLIED = False


def apply_theme() -> None:
    """Set matplotlib rcParams to the LinkedStein dark theme.

    Safe to call multiple times — parameters are set idempotently.
    """
    global _THEME_APPLIED  # noqa: PLW0603
    if _THEME_APPLIED:
        return
    _THEME_APPLIED = True

    rc = mpl.rcParams
    rc["figure.facecolor"] = C_BG_FIGURE
    rc["axes.facecolor"] = C_BG_AXES
    rc["axes.edgecolor"] = C_GRID
    rc["axes.labelcolor"] = C_TEXT
    rc["axes.titlecolor"] = C_TEXT
    rc["axes.prop_cycle"] = PROP_CYCLE
    rc["axes.grid"] = True
    rc["grid.color"] = C_GRID
    rc["grid.alpha"] = 0.25
    rc["xtick.color"] = C_TICK
    rc["ytick.color"] = C_TICK
    rc["text.color"] = C_TEXT
    rc["legend.facecolor"] = PRUSSIAN_BLUE
    rc["legend.edgecolor"] = DUSK_BLUE
    rc["legend.labelcolor"] = C_TEXT
    rc["figure.edgecolor"] = C_BG_FIGURE
    rc["savefig.facecolor"] = C_BG_FIGURE
    rc["savefig.edgecolor"] = C_BG_FIGURE

    plt.style.use("dark_background")
    for k, v in {
        "figure.facecolor": C_BG_FIGURE,
        "axes.facecolor": C_BG_AXES,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_TEXT,
        "axes.titlecolor": C_TEXT,
        "axes.prop_cycle": PROP_CYCLE,
        "grid.color": C_GRID,
        "grid.alpha": 0.25,
        "xtick.color": C_TICK,
        "ytick.color": C_TICK,
        "text.color": C_TEXT,
        "legend.facecolor": PRUSSIAN_BLUE,
        "legend.edgecolor": DUSK_BLUE,
        "legend.labelcolor": C_TEXT,
        "savefig.facecolor": C_BG_FIGURE,
        "savefig.edgecolor": C_BG_FIGURE,
    }.items():
        mpl.rcParams[k] = v
