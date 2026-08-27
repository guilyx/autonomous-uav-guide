# Erwin Lejeune - 2026-08-27
"""Controllers that treat the radio network as part of the plant.

Two problems that look similar and are not:

* **Connectivity maintenance** -- the fleet has a task, and the network must
  survive it. Connectivity is a *constraint*.
* **Relay coverage** -- the fleet's task *is* the network. Spread to cover
  ground, but every agent must keep a multi-hop path home, so coverage and
  connectivity pull against each other directly. That trade-off is provably
  NP-hard in general, so what follows is a gradient heuristic, not an
  optimum, and it is worth being honest about which one you have.

References:
- L. Sabattini et al., "Decentralized connectivity maintenance for
  cooperative control of mobile robotic systems," IJRR, 2013.
  DOI: 10.1177/0278364913499085
- P. Ghassemi & S. Chowdhury, "Multi-robot task allocation in disaster
  response," Robotics and Autonomous Systems, 2022.
- J. Scherer & B. Rinner, "Long-term area coverage and radio relay
  positioning using swarms of UAVs," arXiv:1810.12383, 2018.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .graph import algebraic_connectivity, connectivity_gradient, hop_counts
from .radio import LinkModel

__all__ = ["ConnectivityController", "RelayCoverageController"]


class ConnectivityController:
    """Keep algebraic connectivity above a floor while the fleet does its job.

    The connectivity effort is scaled by a barrier potential in λ₂ rather
    than a fixed gain. A fixed gain has to be tuned for the worst case and
    then distorts the task everywhere else; the barrier is nearly silent
    while the network is healthy and grows without bound as λ₂ approaches
    the floor, so the task is only overridden when it is actually about to
    break the mesh.

    Args:
        link: Link model defining the weighted graph.
        lambda_min: Floor on λ₂. The potential diverges here.
        gain: Overall strength of the connectivity effort.
        max_force: Saturation, so the divergence cannot produce an
            unflyable command.
    """

    def __init__(
        self,
        link: LinkModel,
        lambda_min: float = 0.15,
        gain: float = 12.0,
        max_force: float = 6.0,
    ) -> None:
        if lambda_min <= 0.0:
            raise ValueError(f"lambda_min must be positive, got {lambda_min}")
        if gain <= 0.0:
            raise ValueError(f"gain must be positive, got {gain}")
        self.link = link
        self.lambda_min = float(lambda_min)
        self.gain = float(gain)
        self.max_force = float(max_force)

    def connectivity_value(self, positions: NDArray[np.floating]) -> float:
        return algebraic_connectivity(self.link.weights(positions))

    def forces(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Connectivity-restoring acceleration for each agent."""
        p = np.asarray(positions, dtype=float)
        if len(p) < 2:
            return np.zeros_like(p)

        lam = self.connectivity_value(p)
        slack = lam - self.lambda_min
        # Below the floor the network is already fragmenting: push at full
        # strength rather than evaluating a potential that has gone negative.
        scale = self.gain / max(slack, 1e-3) ** 2 if slack > 0.0 else self.gain / 1e-6

        force = scale * connectivity_gradient(p, self.link)
        norms = np.linalg.norm(force, axis=1, keepdims=True)
        too_big = norms > self.max_force
        return np.where(too_big, force / np.maximum(norms, 1e-12) * self.max_force, force)


class RelayCoverageController:
    """Spread to cover ground while every agent keeps a path to the base.

    Three terms, and the interesting one is the third:

    * **spread** -- mutual repulsion, which is what produces coverage.
    * **anchor** -- a pull outward from the base, so the fleet expands
      rather than settling into a comfortable ball around it.
    * **tether** -- connectivity effort, applied *per agent and scaled by
      hop count*. An agent three hops out is far more likely to be the one
      that severs the chain than one sitting next to the base, so uniform
      connectivity effort wastes it on agents that were never at risk.

    The base station is agent 0 and is assumed stationary; its force is
    zeroed so callers can integrate the whole fleet uniformly.
    """

    def __init__(
        self,
        link: LinkModel,
        spread_gain: float = 40.0,
        anchor_gain: float = 0.35,
        tether_gain: float = 22.0,
        link_threshold: float = 0.5,
        max_force: float = 6.0,
    ) -> None:
        self.link = link
        self.spread_gain = float(spread_gain)
        self.anchor_gain = float(anchor_gain)
        self.tether_gain = float(tether_gain)
        self.link_threshold = float(link_threshold)
        self.max_force = float(max_force)

    def forces(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        p = np.asarray(positions, dtype=float)
        n = len(p)
        out = np.zeros_like(p)
        if n < 2:
            return out

        weights = self.link.weights(p)
        hops = hop_counts(weights, source=0, threshold=self.link_threshold)

        diff = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        safe = np.maximum(dist, 1e-6)

        # Spread: inverse-square repulsion, which falls away fast enough that
        # distant agents do not keep pushing each other outward forever.
        repel = self.spread_gain / safe**2
        np.fill_diagonal(repel, 0.0)
        out += np.einsum("ij,ijk->ik", repel / safe, diff)

        # Anchor: outward from the base, so coverage grows.
        radial = p - p[0]
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        out += self.anchor_gain * radial / np.maximum(radial_norm, 1e-6)

        # Tether: connectivity gradient, weighted by how exposed each agent
        # is. Unreachable agents get the strongest pull of all.
        grad = connectivity_gradient(p, self.link)
        exposure = np.where(np.isinf(hops), float(n), hops).reshape(-1, 1)
        out += self.tether_gain * exposure * grad

        out[0] = 0.0  # the base station does not fly
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        too_big = norms > self.max_force
        return np.where(too_big, out / np.maximum(norms, 1e-12) * self.max_force, out)

    def coverage_fraction(
        self,
        positions: NDArray[np.floating],
        bounds: tuple[float, float],
        sensing_radius: float,
        resolution: float = 2.0,
    ) -> float:
        """Fraction of the area within *sensing_radius* of a connected agent.

        Agents that have lost their path to the base are excluded: ground
        watched by an aircraft that cannot report it is not covered, and
        counting it is the easiest way to make a relay controller look far
        better than it is.
        """
        p = np.asarray(positions, dtype=float)
        weights = self.link.weights(p)
        hops = hop_counts(weights, source=0, threshold=self.link_threshold)
        live = p[np.isfinite(hops)]
        if len(live) == 0:
            return 0.0

        lo, hi = bounds
        axis = np.arange(lo, hi, resolution)
        gx, gy = np.meshgrid(axis, axis)
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        d = np.linalg.norm(cells[:, None, :] - live[None, :, :2], axis=2)
        return float(np.mean(np.min(d, axis=1) <= sensing_radius))
