# Erwin Lejeune - 2026-02-15
"""Reusable data subplot helpers for simulation animations.

Provides functions that create and update live data traces alongside the
3D/2D drone views — e.g. setpoint vs actual, cross-track error,
covariance bounds, velocity profiles.

Typical usage inside a simulation ``update(frame)`` callback::

    update_position_panel(ax, times[:frame], states[:frame], setpoints[:frame])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def setup_position_panel(ax: Axes) -> dict[str, Any]:
    """Prepare axes for position setpoint-vs-actual plot.

    Returns artist dict to pass to :func:`update_position_panel`.
    """
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel("Position [m]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Position Tracking", fontsize=8)
    (l_x,) = ax.plot([], [], "r-", lw=0.8, label="x")
    (l_y,) = ax.plot([], [], "g-", lw=0.8, label="y")
    (l_z,) = ax.plot([], [], "b-", lw=0.8, label="z")
    (l_xs,) = ax.plot([], [], "r--", lw=0.6, alpha=0.6)
    (l_ys,) = ax.plot([], [], "g--", lw=0.6, alpha=0.6)
    (l_zs,) = ax.plot([], [], "b--", lw=0.6, alpha=0.6)
    ax.legend(fontsize=5, loc="upper right", ncol=3)
    return {"ax": ax, "lines": [l_x, l_y, l_z, l_xs, l_ys, l_zs]}


def update_position_panel(
    artists: dict[str, Any],
    t: NDArray[np.floating],
    actual: NDArray[np.floating],
    setpoint: NDArray[np.floating] | None = None,
) -> None:
    """Update position panel with data up to current frame.

    Parameters
    ----------
    artists : dict returned by :func:`setup_position_panel`.
    t : 1D array of time values.
    actual : (N, 3) position array.
    setpoint : optional (N, 3) setpoint array.
    """
    lines = artists["lines"]
    ax = artists["ax"]
    for i, ln in enumerate(lines[:3]):
        ln.set_data(t, actual[:, i])
    if setpoint is not None:
        for i, ln in enumerate(lines[3:]):
            ln.set_data(t, setpoint[:, i])
    if len(t) > 0:
        ax.set_xlim(0, max(t[-1], 0.1))
        all_vals = actual.flatten()
        if setpoint is not None:
            all_vals = np.concatenate([all_vals, setpoint.flatten()])
        mn, mx = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        margin = max(0.5, (mx - mn) * 0.1)
        ax.set_ylim(mn - margin, mx + margin)


def setup_attitude_panel(ax: Axes) -> dict[str, Any]:
    """Prepare axes for Euler angle plot."""
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel("Angle [rad]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Attitude", fontsize=8)
    (l_phi,) = ax.plot([], [], "r-", lw=0.8, label=r"$\phi$")
    (l_theta,) = ax.plot([], [], "g-", lw=0.8, label=r"$\theta$")
    (l_psi,) = ax.plot([], [], "b-", lw=0.8, label=r"$\psi$")
    ax.legend(fontsize=5, loc="upper right", ncol=3)
    return {"ax": ax, "lines": [l_phi, l_theta, l_psi]}


def update_attitude_panel(
    artists: dict[str, Any],
    t: NDArray[np.floating],
    euler: NDArray[np.floating],
) -> None:
    """Update attitude panel."""
    lines = artists["lines"]
    ax = artists["ax"]
    for i, ln in enumerate(lines):
        ln.set_data(t, euler[:, i])
    if len(t) > 0:
        ax.set_xlim(0, max(t[-1], 0.1))
        mn, mx = float(np.nanmin(euler)), float(np.nanmax(euler))
        margin = max(0.1, (mx - mn) * 0.1)
        ax.set_ylim(mn - margin, mx + margin)


def setup_error_panel(ax: Axes, ylabel: str = "Error [m]") -> dict[str, Any]:
    """Prepare axes for a single error trace (e.g. cross-track error)."""
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Tracking Error", fontsize=8)
    (l_err,) = ax.plot([], [], "m-", lw=0.8)
    return {"ax": ax, "line": l_err}


def update_error_panel(
    artists: dict[str, Any],
    t: NDArray[np.floating],
    error: NDArray[np.floating],
) -> None:
    """Update single-trace error panel."""
    ax = artists["ax"]
    artists["line"].set_data(t, error)
    if len(t) > 0:
        ax.set_xlim(0, max(t[-1], 0.1))
        mx = float(np.nanmax(np.abs(error)))
        ax.set_ylim(0, max(mx * 1.2, 0.1))


def setup_estimation_panel(ax: Axes) -> dict[str, Any]:
    """Prepare axes for estimation filter output (estimated vs true + 2-sigma)."""
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel("State", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Estimation", fontsize=8)
    (l_true,) = ax.plot([], [], "k-", lw=0.8, label="True")
    (l_est,) = ax.plot([], [], "r-", lw=0.8, label="Est")
    fill = ax.fill_between([], [], [], color="r", alpha=0.15, label=r"$2\sigma$")
    ax.legend(fontsize=5, loc="upper right", ncol=3)
    return {"ax": ax, "l_true": l_true, "l_est": l_est, "fill": fill}


def update_estimation_panel(
    artists: dict[str, Any],
    t: NDArray[np.floating],
    true_val: NDArray[np.floating],
    est_val: NDArray[np.floating],
    sigma: NDArray[np.floating] | None = None,
) -> None:
    """Update estimation panel with true, estimated, and optional 2-sigma bounds."""
    ax = artists["ax"]
    artists["l_true"].set_data(t, true_val)
    artists["l_est"].set_data(t, est_val)
    if sigma is not None and len(t) > 1:
        artists["fill"].remove()
        artists["fill"] = ax.fill_between(
            t, est_val - 2 * sigma, est_val + 2 * sigma, color="r", alpha=0.15
        )
    if len(t) > 0:
        ax.set_xlim(0, max(t[-1], 0.1))
        all_v = np.concatenate([true_val, est_val])
        mn, mx = float(np.nanmin(all_v)), float(np.nanmax(all_v))
        margin = max(0.5, (mx - mn) * 0.1)
        ax.set_ylim(mn - margin, mx + margin)


def setup_velocity_panel(ax: Axes) -> dict[str, Any]:
    """Prepare axes for velocity profile."""
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel("Velocity [m/s]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Velocity", fontsize=8)
    (l_vx,) = ax.plot([], [], "r-", lw=0.8, label="vx")
    (l_vy,) = ax.plot([], [], "g-", lw=0.8, label="vy")
    (l_vz,) = ax.plot([], [], "b-", lw=0.8, label="vz")
    ax.legend(fontsize=5, loc="upper right", ncol=3)
    return {"ax": ax, "lines": [l_vx, l_vy, l_vz]}


def update_velocity_panel(
    artists: dict[str, Any],
    t: NDArray[np.floating],
    velocity: NDArray[np.floating],
) -> None:
    """Update velocity panel."""
    lines = artists["lines"]
    ax = artists["ax"]
    for i, ln in enumerate(lines):
        ln.set_data(t, velocity[:, i])
    if len(t) > 0:
        ax.set_xlim(0, max(t[-1], 0.1))
        mn, mx = float(np.nanmin(velocity)), float(np.nanmax(velocity))
        margin = max(0.5, (mx - mn) * 0.1)
        ax.set_ylim(mn - margin, mx + margin)


def setup_thrust_panel(ax: Axes) -> dict[str, Any]:
    """Prepare axes for thrust output."""
    ax.set_xlabel("Time [s]", fontsize=7)
    ax.set_ylabel("Thrust [N]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("Thrust", fontsize=8)
    (l_t,) = ax.plot([], [], "k-", lw=0.8)
    (l_hover,) = ax.plot([], [], "k--", lw=0.5, alpha=0.5, label="hover")
    ax.legend(fontsize=5, loc="upper right")
    return {"ax": ax, "l_thrust": l_t, "l_hover": l_hover}


def update_thrust_panel(
    artists: dict[str, Any],
    t: NDArray[np.floating],
    thrust: NDArray[np.floating],
    hover_thrust: float | None = None,
) -> None:
    """Update thrust panel."""
    ax = artists["ax"]
    artists["l_thrust"].set_data(t, thrust)
    if hover_thrust is not None and len(t) > 0:
        artists["l_hover"].set_data([0, t[-1]], [hover_thrust, hover_thrust])
    if len(t) > 0:
        ax.set_xlim(0, max(t[-1], 0.1))
        mx = float(np.nanmax(thrust))
        ax.set_ylim(0, max(mx * 1.2, 1.0))
