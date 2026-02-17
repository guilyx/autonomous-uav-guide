# Erwin Lejeune - 2026-02-15
"""Feedback-linearisation tracker on a circular reference.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011, Sec. IV. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.models.quadrotor import Quadrotor
from quadrotor_sim.tracking.feedback_linearisation import FeedbackLinearisationTracker
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.array([1, 0, 1.0]))
    hf = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hf))
    tracker = FeedbackLinearisationTracker(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )
    dt, dur = 0.002, 8.0
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    refs = np.zeros((steps, 3))
    for i in range(steps):
        t = i * dt
        rp = np.array([np.cos(0.5 * t), np.sin(0.5 * t), 1 + 0.2 * np.sin(t)])
        rv = np.array([-0.5 * np.sin(0.5 * t), 0.5 * np.cos(0.5 * t), 0.2 * np.cos(t)])
        quad.step(tracker.compute(quad.state, rp, rv, np.zeros(3)), dt)
        states[i] = quad.state
        refs[i] = rp
    pos = states[:, :3]
    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))
    anim = SimAnimator("feedback_linearisation", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Feedback Linearisation Tracking")
    ax.plot(refs[:, 0], refs[:, 1], refs[:, 2], "r--", lw=1, alpha=0.5, label="Ref")
    (trail,) = ax.plot([], [], [], "b-", lw=1.2, alpha=0.7)
    (dot,) = ax.plot([], [], [], "ko", ms=5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 1.5)
    ax.legend(fontsize=8)

    def update(f):
        k = idx[f]
        trail.set_data(pos[:k, 0], pos[:k, 1])
        trail.set_3d_properties(pos[:k, 2])
        dot.set_data([pos[k, 0]], [pos[k, 1]])
        dot.set_3d_properties([pos[k, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
