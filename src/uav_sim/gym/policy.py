# Erwin Lejeune - 2026-02-18
"""Small deterministic policies, implemented in NumPy.

Deliberately dependency-free. Training a drone to fly should not require a
deep-learning stack — these tasks are low-dimensional enough that a two
hidden layer MLP driven by an evolutionary search solves them, and being
able to run ``uav-sim train hover`` on a fresh ``pip install`` matters more
here than squeezing out the last few percent of return.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["MLPPolicy", "RunningNormalizer"]


@dataclass
class RunningNormalizer:
    """Online mean/variance tracker for observations (Welford's algorithm).

    Observation components here span wildly different scales — metres,
    radians, rad/s — and an unnormalised policy spends its early training
    budget just learning to ignore whichever input happens to be largest.
    """

    size: int
    mean: NDArray[np.floating] = None  # type: ignore[assignment]
    var: NDArray[np.floating] = None  # type: ignore[assignment]
    count: float = 1e-4

    def __post_init__(self) -> None:
        if self.mean is None:
            self.mean = np.zeros(self.size)
        if self.var is None:
            self.var = np.ones(self.size)

    def update(self, batch: NDArray[np.floating]) -> None:
        batch = np.atleast_2d(batch)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + np.square(delta) * self.count * batch_count / total) / total
        self.count = total

    def __call__(self, observation: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.clip((observation - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


class MLPPolicy:
    """Deterministic tanh-MLP mapping observations to actions in ``[-1, 1]``.

    Parameters are stored flat so an evolutionary optimiser can treat the
    whole network as a single vector.

    Examples
    --------
    >>> policy = MLPPolicy(observation_size=15, action_size=4, seed=0)
    >>> action = policy.act(np.zeros(15))
    >>> action.shape
    (4,)
    >>> bool(np.all(np.abs(action) <= 1.0))
    True
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        seed: int | None = None,
        normalize: bool = False,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_sizes = tuple(hidden_sizes)
        # Off by default. These environments already emit observations in
        # sensible units — metres, m/s, rotation-matrix entries, rad/s — so
        # whitening buys little, and fitting the statistics on early
        # crash-heavy rollouts actively hurts: the variance of the position
        # error is dominated by the metres-wide excursions of a failing
        # policy, so the sub-metre errors a good policy cares about get
        # divided down into noise.
        self.normalize = normalize
        self.normalizer = RunningNormalizer(observation_size)

        self._shapes: list[tuple[int, int]] = []
        sizes = (observation_size, *self.hidden_sizes, action_size)
        for a, b in zip(sizes[:-1], sizes[1:]):
            self._shapes.append((a, b))

        rng = np.random.default_rng(seed)
        chunks = []
        for index, (fan_in, fan_out) in enumerate(self._shapes):
            if index == len(self._shapes) - 1:
                # Zero the output layer so the initial policy emits exactly
                # zero actions. The environments centre their action spaces
                # on equilibrium — hover thrust for a quadrotor, trim for a
                # fixed wing — so a zero action is a *flying* action.
                #
                # This is the difference between a search that works and one
                # that does not. From a random output layer every candidate
                # saturates tanh, commands full torque and tumbles within a
                # second, so every candidate scores the same and the
                # optimiser has no signal to follow.
                weights = np.zeros((fan_in, fan_out))
            else:
                # Xavier scaling keeps hidden activations off the tanh rails.
                weights = rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
            chunks.append(weights.ravel())
            chunks.append(np.zeros(fan_out))
        self.parameters = np.concatenate(chunks)

    @property
    def parameter_count(self) -> int:
        return int(self.parameters.size)

    def _layers(self, parameters: NDArray[np.floating]):
        offset = 0
        for fan_in, fan_out in self._shapes:
            weight_size = fan_in * fan_out
            weights = parameters[offset : offset + weight_size].reshape(fan_in, fan_out)
            offset += weight_size
            bias = parameters[offset : offset + fan_out]
            offset += fan_out
            yield weights, bias

    def act(
        self,
        observation: NDArray[np.floating],
        parameters: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Return an action for one observation."""
        x = np.asarray(observation, dtype=np.float64)
        if self.normalize:
            x = self.normalizer(x)
        theta = self.parameters if parameters is None else parameters
        layers = list(self._layers(theta))
        for index, (weights, bias) in enumerate(layers):
            x = x @ weights + bias
            # tanh on the output layer too — it *is* the action squashing.
            x = np.tanh(x) if index < len(layers) - 1 else np.tanh(x)
        return x

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Write the policy to a portable ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            parameters=self.parameters,
            normalizer_mean=self.normalizer.mean,
            normalizer_var=self.normalizer.var,
            normalizer_count=self.normalizer.count,
            meta=json.dumps(
                {
                    "observation_size": self.observation_size,
                    "action_size": self.action_size,
                    "hidden_sizes": list(self.hidden_sizes),
                    "normalize": self.normalize,
                }
            ),
        )
        return path.with_suffix(".npz") if path.suffix != ".npz" else path

    @classmethod
    def load(cls, path: str | Path) -> "MLPPolicy":
        """Load a policy previously written by :meth:`save`."""
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        policy = cls(
            observation_size=meta["observation_size"],
            action_size=meta["action_size"],
            hidden_sizes=tuple(meta["hidden_sizes"]),
            normalize=meta.get("normalize", False),
        )
        policy.parameters = data["parameters"]
        policy.normalizer.mean = data["normalizer_mean"]
        policy.normalizer.var = data["normalizer_var"]
        policy.normalizer.count = float(data["normalizer_count"])
        return policy
