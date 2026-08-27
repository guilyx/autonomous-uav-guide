# Erwin Lejeune - 2026-02-18
"""Rendering for reinforcement-learning episodes.

Draws the flown trajectories against the task's goal so a trained policy
can be inspected rather than just scored. Matplotlib only — no simulator
window, no OpenGL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["render_episode", "render_learning_curve"]

# Shared with the documentation site so plots and pages read as one design.
_INK = "#0b1220"
_GRID = "#1e293b"
_TEXT = "#e2e8f0"
_ACCENT = "#38bdf8"
_GOAL = "#fbbf24"
_TRACE = ["#38bdf8", "#2dd4bf", "#a78bfa", "#f472b6", "#fb923c"]


def render_episode(
    env,
    trajectories: list[NDArray[np.floating]],
    path: str | Path,
    *,
    title: str = "policy rollout",
    fps: int = 25,
) -> Path:
    """Animate flown trajectories to a GIF.

    Parameters
    ----------
    env
        The environment they were flown in, used for the goal and bounds.
    trajectories
        One ``(steps, 3)`` array per episode.
    path
        Output file. A ``.gif`` suffix is added when missing.
    title
        Figure title.
    fps
        Animation frame rate.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = Path(path)
    if path.suffix != ".gif":
        path = path.with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.2, 5.4), facecolor=_INK)
    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor(_INK)

    stacked = np.vstack([t for t in trajectories if len(t)])
    span = max(float(np.ptp(stacked[:, :2])), 4.0) * 0.6
    centre = stacked[:, :2].mean(axis=0)
    axes.set_xlim(centre[0] - span, centre[0] + span)
    axes.set_ylim(centre[1] - span, centre[1] + span)
    axes.set_zlim(0.0, max(float(stacked[:, 2].max()) * 1.15, 2.0))

    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((0.04, 0.07, 0.13, 1.0))
        axis._axinfo["grid"]["color"] = _GRID
    axes.tick_params(colors=_TEXT, labelsize=7)
    axes.set_xlabel("x [m]", color=_TEXT, fontsize=8)
    axes.set_ylabel("y [m]", color=_TEXT, fontsize=8)
    axes.set_zlabel("z [m]", color=_TEXT, fontsize=8)
    axes.set_title(title, color=_TEXT, fontsize=11, pad=14)

    goal = np.asarray(getattr(env, "goal", np.zeros(3)))
    axes.scatter(*goal, color=_GOAL, s=70, marker="*", label="goal", depthshade=False)

    lines, heads = [], []
    for index in range(len(trajectories)):
        colour = _TRACE[index % len(_TRACE)]
        (line,) = axes.plot([], [], [], color=colour, lw=1.4, alpha=0.9)
        (head,) = axes.plot([], [], [], "o", color=colour, ms=5)
        lines.append(line)
        heads.append(head)

    axes.legend(loc="upper right", fontsize=7, facecolor=_INK, edgecolor=_GRID, labelcolor=_TEXT)
    frames = max(len(t) for t in trajectories)
    stride = max(1, frames // 240)
    frame_indices = list(range(0, frames, stride))

    def update(frame_number: int):
        cursor = frame_indices[min(frame_number, len(frame_indices) - 1)]
        for trajectory, line, head in zip(trajectories, lines, heads):
            visible = trajectory[: min(cursor + 1, len(trajectory))]
            if len(visible) == 0:
                continue
            line.set_data(visible[:, 0], visible[:, 1])
            line.set_3d_properties(visible[:, 2])
            head.set_data([visible[-1, 0]], [visible[-1, 1]])
            head.set_3d_properties([visible[-1, 2]])
        axes.view_init(elev=22, azim=-60 + 0.25 * cursor)
        return [*lines, *heads]

    animation = FuncAnimation(figure, update, frames=len(frame_indices), blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100, savefig_kwargs={"facecolor": _INK})
    plt.close(figure)
    return path


def render_learning_curve(
    history: list[dict[str, float]],
    path: str | Path,
    *,
    title: str = "learning curve",
) -> Path:
    """Plot per-generation returns to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    generations = [row["generation"] for row in history]
    figure, axes = plt.subplots(figsize=(7.0, 4.0), facecolor=_INK)
    axes.set_facecolor(_INK)
    axes.plot(
        generations,
        [r["mean_return"] for r in history],
        color=_GRID,
        lw=1.2,
        label="population mean",
    )
    axes.plot(
        generations,
        [r["elite_return"] for r in history],
        color=_ACCENT,
        lw=1.8,
        label="elite mean",
    )
    axes.plot(
        generations,
        [r["mean_policy_return"] for r in history],
        color=_GOAL,
        lw=2.0,
        label="returned policy",
    )

    axes.set_xlabel("generation", color=_TEXT)
    axes.set_ylabel("episode return", color=_TEXT)
    axes.set_title(title, color=_TEXT, fontsize=12)
    axes.tick_params(colors=_TEXT)
    for spine in axes.spines.values():
        spine.set_color(_GRID)
    axes.grid(True, color=_GRID, alpha=0.4, lw=0.6)
    axes.legend(facecolor=_INK, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=140, facecolor=_INK)
    plt.close(figure)
    return path
