# Erwin Lejeune - 2026-02-15
"""Reusable animation recorder for all quadrotor simulations.

Every simulation script creates a `SimAnimator`, adds frames via its
``update`` callback, and calls :meth:`save` to write a GIF. The class
always uses the *Agg* backend so it works headless on CI and remote
machines.

Typical usage
-------------
>>> anim = SimAnimator("pid_hover", out_dir=Path(__file__).parent)
>>> fig, ax = anim.figure_3d("PID Hover")
>>> # pre-compute data …
>>> def draw(frame):
...     # update artists for this frame
...     pass
>>> anim.animate(draw, n_frames=200)
>>> anim.save()                          # → <out_dir>/pid_hover.gif
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

# Force non-interactive backend so we never hit the FigureCanvasAgg warning.
matplotlib.use("Agg")

# ── defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_FPS = 25
_DEFAULT_DPI = 90
_DEFAULT_FIGSIZE_3D = (7, 6)
_DEFAULT_FIGSIZE_2D = (8, 5)


class SimAnimator:
    """Create, animate, and save a simulation GIF.

    Parameters
    ----------
    name:
        Filename stem for the output GIF (e.g. ``"pid_hover"``).
    out_dir:
        Directory where ``<name>.gif`` is written.
    fps:
        Frames per second in the GIF.
    dpi:
        Dots per inch for each frame.
    """

    def __init__(
        self,
        name: str,
        out_dir: Path | str = ".",
        fps: int = _DEFAULT_FPS,
        dpi: int = _DEFAULT_DPI,
    ) -> None:
        self.name = name
        self.out_dir = Path(out_dir)
        self.fps = fps
        self.dpi = dpi
        self._fig: Figure | None = None
        self._anim: animation.FuncAnimation | None = None

    # ── figure factories ──────────────────────────────────────────────────

    def figure_3d(
        self,
        title: str = "",
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE_3D,
    ) -> tuple[Figure, Axes3D]:
        """Create a 3D figure and return ``(fig, ax)``."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if title:
            ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        self._fig = fig
        return fig, ax

    def figure_2d(
        self,
        title: str = "",
        nrows: int = 1,
        figsize: tuple[float, float] | None = None,
        sharex: bool = False,
    ) -> tuple[Figure, np.ndarray]:
        """Create a 2D figure with *nrows* subplots.

        Returns ``(fig, axes)`` where ``axes`` is always an ndarray.
        """
        if figsize is None:
            figsize = (_DEFAULT_FIGSIZE_2D[0], max(4, 2.5 * nrows))
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=sharex, squeeze=False)
        axes = axes.ravel()
        if title:
            fig.suptitle(title)
        self._fig = fig
        return fig, axes

    # ── 3D drawing helpers ────────────────────────────────────────────────

    @staticmethod
    def draw_sphere(
        ax: Axes3D,
        centre: NDArray[np.floating],
        radius: float,
        color: str = "red",
        alpha: float = 0.25,
    ) -> None:
        """Draw a semi-transparent sphere obstacle."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=alpha, color=color)

    @staticmethod
    def draw_quadrotor(
        ax: Axes3D,
        position: NDArray[np.floating],
        R: NDArray[np.floating],
        arm_length: float = 0.0397,
        scale: float = 10.0,
    ) -> None:
        """Draw a quadrotor frame at the given pose."""
        L = arm_length * scale
        s = L / np.sqrt(2.0)
        motors_body = np.array([[s, s, 0], [s, -s, 0], [-s, -s, 0], [-s, s, 0]]).T
        motors_world = R @ motors_body + position.reshape(3, 1)
        colors = ["r", "b", "r", "b"]
        for i in range(4):
            ax.plot(
                [position[0], motors_world[0, i]],
                [position[1], motors_world[1, i]],
                [position[2], motors_world[2, i]],
                color=colors[i],
                linewidth=2,
            )
            ax.scatter(*motors_world[:, i], color=colors[i], s=20)
        ax.scatter(*position, color="k", s=30, marker="o")

    @staticmethod
    def set_equal_3d(ax: Axes3D, data: NDArray[np.floating], pad: float = 1.0) -> None:
        """Set equal-aspect 3D axis limits around *data* ``(N, 3)``."""
        if data.ndim == 3:
            data = data.reshape(-1, 3)
        lo = data.min(axis=0) - pad
        hi = data.max(axis=0) + pad
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])

    # ── animation ─────────────────────────────────────────────────────────

    def animate(
        self,
        update_fn,
        n_frames: int,
        interval_ms: int | None = None,
    ) -> animation.FuncAnimation:
        """Build a `FuncAnimation` and store it.

        Parameters
        ----------
        update_fn:
            Callable ``update(frame_index) -> iterable_of_artists``.
        n_frames:
            Total number of animation frames.
        interval_ms:
            Delay between frames in ms.  Defaults to ``1000 / fps``.
        """
        if self._fig is None:
            raise RuntimeError("Create a figure first (figure_3d / figure_2d).")
        if interval_ms is None:
            interval_ms = int(1000 / self.fps)
        self._anim = animation.FuncAnimation(
            self._fig, update_fn, frames=n_frames, interval=interval_ms, blit=False
        )
        return self._anim

    def save(self) -> Path:
        """Write the animation as a GIF and return the output path."""
        if self._anim is None:
            raise RuntimeError("Call animate() before save().")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = self.out_dir / f"{self.name}.gif"
        writer = animation.PillowWriter(fps=self.fps)
        self._anim.save(str(out), writer=writer, dpi=self.dpi)
        plt.close(self._fig)
        print(f"  GIF saved → {out}")
        return out
