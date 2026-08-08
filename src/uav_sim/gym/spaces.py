# Erwin Lejeune - 2026-02-18
"""Minimal observation and action spaces.

These mirror the parts of :mod:`gymnasium.spaces` the environments in this
package actually use. Re-implementing them keeps ``uav_sim`` installable
with nothing but NumPy, while staying duck-type compatible with Gymnasium
so the same environments work under either.

If Gymnasium *is* installed, :func:`uav_sim.gym.registry.register_gymnasium`
converts these into the real thing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["Box"]


class Box:
    """A closed box in :math:`\\mathbb{R}^n`.

    Parameters
    ----------
    low, high
        Bounds, either scalars broadcast to ``shape`` or full arrays.
    shape
        Required only when ``low`` and ``high`` are scalars.

    Examples
    --------
    >>> space = Box(-1.0, 1.0, shape=(4,))
    >>> space.shape
    (4,)
    >>> bool(space.contains(np.zeros(4)))
    True
    """

    def __init__(
        self,
        low: float | NDArray[np.floating],
        high: float | NDArray[np.floating],
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float32,
    ) -> None:
        if shape is None:
            low_array = np.asarray(low, dtype=dtype)
            high_array = np.asarray(high, dtype=dtype)
            if low_array.shape != high_array.shape:
                raise ValueError(
                    f"low and high must have the same shape, got "
                    f"{low_array.shape} and {high_array.shape}"
                )
            shape = low_array.shape
        else:
            low_array = np.full(shape, low, dtype=dtype)
            high_array = np.full(shape, high, dtype=dtype)

        if np.any(low_array > high_array):
            raise ValueError("Every low bound must be <= its matching high bound.")

        self.low = low_array
        self.high = high_array
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.floating]:
        """Draw a uniform sample, with infinite bounds clipped to a usable range."""
        generator = rng if rng is not None else np.random.default_rng()
        low = np.clip(self.low, -1e3, 1e3)
        high = np.clip(self.high, -1e3, 1e3)
        return generator.uniform(low, high).astype(self.dtype)

    def contains(self, value: NDArray[np.floating]) -> bool:
        value = np.asarray(value)
        return bool(
            value.shape == self.shape and np.all(value >= self.low) and np.all(value <= self.high)
        )

    def clip(self, value: NDArray[np.floating]) -> NDArray[np.floating]:
        """Project a value into the box."""
        return np.clip(np.asarray(value, dtype=self.dtype), self.low, self.high)

    def __repr__(self) -> str:
        return f"Box({self.low.min():.3g}, {self.high.max():.3g}, {self.shape})"
