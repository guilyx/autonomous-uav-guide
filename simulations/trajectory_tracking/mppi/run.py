# Erwin Lejeune - 2026-02-15
"""MPPI sampling-based tracker on a step reference.

Reference: G. Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning," ICRA, 2017. DOI: 10.1109/ICRA.2017.7989202
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.trajectory_tracking.mppi import MPPITracker
from uav_sim.visualization import SimAnimator


def _dyn(x, u, dt):
    pos, vel = x[:3], x[3:6]
    nv = vel + u * dt
    np_ = pos + nv * dt
    return np.concatenate([np_, nv])


def _cost(x, _u, ref):
    if ref is None:
        return 0.0
    return float(np.sum((x[:3] - ref[:3]) ** 2) + 0.1 * np.sum((x[3:6] - ref[3:6]) ** 2))


def main() -> None:
    tgt = np.array([2, 2, 1, 0, 0, 0.0])
    tracker = MPPITracker(
        state_dim=6,
        control_dim=3,
        horizon=20,
        num_samples=256,
        lambda_=1.0,
        control_std=np.array([2, 2, 2.0]),
        dynamics=_dyn,
        cost_fn=_cost,
        dt=0.05,
    )
    state = np.zeros(6)
    dt, N = 0.05, 200
    hist = np.zeros((N, 6))
    for i in range(N):
        u = tracker.compute(state, reference=tgt, seed=i)
        state = _dyn(state, u, dt)
        hist[i] = state
    t = np.arange(N) * dt
    skip = max(1, N // 200)
    idx = list(range(0, N, skip))
    anim = SimAnimator("mppi", out_dir=Path(__file__).parent)
    _, axes = anim.figure_2d("MPPI Tracker — Step Response", nrows=2, sharex=True)
    lbl = ["x", "y", "z"]
    lp = [axes[0].plot([], [], label=lb)[0] for lb in lbl]
    for j, v in enumerate(tgt[:3]):
        axes[0].axhline(v, color=f"C{j}", ls="--", alpha=0.3)
    axes[0].set_xlim(0, N * dt)
    axes[0].set_ylim(-0.5, 3)
    axes[0].set_ylabel("Pos [m]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    lv = [axes[1].plot([], [], label=f"v{lb}")[0] for lb in lbl]
    axes[1].set_xlim(0, N * dt)
    axes[1].set_ylim(-2, 3)
    axes[1].set_ylabel("Vel [m/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    def update(f):
        k = idx[f]
        for j, line in enumerate(lp):
            line.set_data(t[:k], hist[:k, j])
        for j, line in enumerate(lv):
            line.set_data(t[:k], hist[:k, 3 + j])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
