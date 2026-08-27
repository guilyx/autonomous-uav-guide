# Erwin Lejeune - 2026-02-22
"""Discrete-time process-noise covariances for constant-velocity models.

A Kalman filter's ``Q`` is the covariance accumulated over **one step**,
so it has to scale with ``dt``.  Writing ``Q = diag([...])`` with fixed
numbers is the single most common way to detune a filter: at 200 Hz a
"reasonable looking" ``0.1`` on the velocity states tells the filter the
velocity random-walks by 0.32 m/s every 5 ms, so it throws the prediction
away and simply echoes the measurement.  Halve the step and the filter
silently becomes twice as noisy again.

Two standard discretisations are provided:

``constant_velocity_q``
    Continuous white-noise acceleration (CWNA).  Unmodelled acceleration
    is a continuous white process of power spectral density ``psd``, so
    the accumulated covariance over a fixed interval is the same however
    finely it is integrated.  Use this when nothing measures the
    acceleration and the filter is coasting on a constant-velocity model.

``constant_acceleration_input_q``
    Piecewise white-noise acceleration (PWNA).  The acceleration is
    *measured* (an accelerometer) and its error is white **per sample**,
    which is exactly the piecewise assumption.

Reference: Y. Bar-Shalom, X.-R. Li, T. Kirubarajan, "Estimation with
Applications to Tracking and Navigation," Wiley, 2001, Section 6.2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["constant_velocity_q", "constant_acceleration_input_q"]


def _block_diag_per_axis(dt: float, block: NDArray[np.floating], dim: int) -> NDArray[np.floating]:
    """Scatter a 2x2 ``[p, v]`` block onto a ``[p (dim), v (dim)]`` state."""
    q = np.zeros((2 * dim, 2 * dim))
    for i in range(dim):
        idx = [i, dim + i]
        q[np.ix_(idx, idx)] = block
    return q


def constant_velocity_q(dt: float, psd: float, dim: int = 3) -> NDArray[np.floating]:
    """``Q`` for a ``[position, velocity]`` state coasting between measurements.

    Continuous white-noise acceleration:

    ``Q = psd · [[dt³/3, dt²/2], [dt²/2, dt]]`` per axis.

    Because the acceleration is modelled as a continuous process, running
    the same filter at 50 Hz or 200 Hz accumulates the same uncertainty
    between measurements — which is what makes the tuning transferable.

    Parameters
    ----------
    dt : filter step [s].
    psd : power spectral density of the unmodelled acceleration
        [(m/s²)²/Hz].  A useful starting point is ``a_max² · 2τ`` where
        ``τ`` is the time over which the true acceleration stays
        correlated; tune it until the reported 1σ brackets the actual
        error.
    dim : number of spatial axes (3 for x/y/z).

    Returns
    -------
    ``(2·dim, 2·dim)`` positive semi-definite covariance, state ordered
    ``[p (dim), v (dim)]``.
    """
    block = psd * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    return _block_diag_per_axis(dt, block, dim)


def constant_acceleration_input_q(
    dt: float,
    sigma_a: float,
    dim: int = 3,
    sigma_bias: float = 0.0,
) -> NDArray[np.floating]:
    """``Q`` for a ``[position, velocity]`` state with a measured acceleration input.

    Piecewise white-noise acceleration: over one step an error
    ``a ~ N(0, sigma_a²)`` in the acceleration that was fed in moves the
    position by ``a dt²/2`` and the velocity by ``a dt``, so

    ``Q = sigma_a² · g gᵀ`` with ``g = [dt²/2, dt]`` per axis.

    Parameters
    ----------
    dt : filter step [s].
    sigma_a : accelerometer white-noise standard deviation [m/s²].
    dim : number of spatial axes.
    sigma_bias : bias random-walk density [m/s²/√s], added as extra
        velocity noise so an unmodelled turn-on bias does not make the
        filter overconfident.  ``0`` disables it.
    """
    g = np.array([0.5 * dt * dt, dt])
    block = sigma_a**2 * np.outer(g, g)
    if sigma_bias > 0.0:
        block = block + np.array([[0.0, 0.0], [0.0, sigma_bias**2 * dt]])
    return _block_diag_per_axis(dt, block, dim)
