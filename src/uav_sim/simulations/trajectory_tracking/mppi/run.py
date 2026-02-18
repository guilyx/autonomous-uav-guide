# Erwin Lejeune - 2026-02-15
"""MPPI trajectory tracking: 3-panel point-mass planning + quadrotor flight.

Phase 1 — Algorithm: MPPI sampling on a point-mass model plans a path to
           the target through the urban environment.
Phase 2 — Platform: quadrotor follows the MPPI-planned path using pure
           pursuit + PID for stable execution.

Reference: G. Williams et al., "Information Theoretic MPC for Model-Based
Reinforcement Learning," ICRA, 2017. DOI: 10.1109/ICRA.2017.7989202
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.environment.obstacles import BoxObstacle
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_tracking.mppi import MPPITracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
START = np.array([3.0, 3.0, 12.0, 0.0, 0.0, 0.0])
GOAL = np.array([27.0, 27.0, 12.0, 0.0, 0.0, 0.0])

_OBS_SPHERES: list[tuple[np.ndarray, float]] = []


def _box_to_sphere(b: BoxObstacle) -> tuple[np.ndarray, float]:
    centre = (b.min_corner + b.max_corner) / 2
    radius = float(np.linalg.norm(b.max_corner - b.min_corner)) / 2
    return (centre, radius)


def _dyn(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    pos, vel = x[:3], x[3:6]
    nv = vel + u * dt
    np_ = pos + nv * dt
    return np.concatenate([np_, nv])


def _cost(x: np.ndarray, _u: np.ndarray, ref: np.ndarray | None) -> float:
    if ref is None:
        return 0.0
    goal_cost = float(np.sum((x[:3] - ref[:3]) ** 2) + 0.1 * np.sum((x[3:6] - ref[3:6]) ** 2))
    obs_cost = 0.0
    for c, r in _OBS_SPHERES:
        dist = float(np.linalg.norm(x[:3] - c))
        if dist < r + 1.0:
            obs_cost += 1e4 * max(0.0, r + 1.0 - dist) ** 2
    return goal_cost + obs_cost


def main() -> None:
    global _OBS_SPHERES  # noqa: PLW0603
    world, buildings = default_world()
    _OBS_SPHERES = [_box_to_sphere(b) for b in buildings]

    tracker = MPPITracker(
        state_dim=6,
        control_dim=3,
        horizon=20,
        num_samples=256,
        lambda_=0.3,
        control_std=np.array([1.5, 1.5, 0.8]),
        dynamics=_dyn,
        cost_fn=_cost,
        dt=0.1,
    )

    # Phase 1: point-mass MPPI planning towards goal
    state = START.copy()
    dt_mppi = 0.1
    max_steps = 400
    mppi_hist: list[np.ndarray] = []
    for i in range(max_steps):
        u = tracker.compute(state, reference=GOAL, seed=i)
        state = _dyn(state, u, dt_mppi)
        mppi_hist.append(state.copy())
        if np.linalg.norm(state[:3] - GOAL[:3]) < 1.5:
            break

    mppi_arr = np.array(mppi_hist)
    mppi_path = mppi_arr[:, :3]
    n_steps = len(mppi_path)

    # Phase 2: quadrotor follows MPPI path using pure pursuit
    quad = Quadrotor()
    quad.reset(position=np.array([START[0], START[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    flight_states = fly_mission(
        quad,
        ctrl,
        mppi_path,
        cruise_alt=12.0,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.0,
        landing_duration=2.0,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation ─────────────────────────────────────────────────────
    mppi_skip = max(1, n_steps // 100)
    mppi_idx = list(range(0, n_steps, mppi_skip))
    fly_skip = max(1, len(flight_pos) // 100)
    fly_idx = list(range(0, len(flight_pos), fly_skip))
    n_mf = len(mppi_idx)
    n_ff = len(fly_idx)
    total = n_mf + n_ff

    viz = ThreePanelViz(title="MPPI Trajectory Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.mark_start_goal(START[:3], GOAL[:3])

    mppi_trail = viz.create_trail_artists(color="cyan")
    fly_trail = viz.create_trail_artists(color="orange")

    title = viz.ax3d.set_title("Phase 1: MPPI Planning")

    anim = SimAnimator("mppi", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < n_mf:
            k = mppi_idx[f]
            viz.update_trail(mppi_trail, mppi_path, k + 1)
            pct = int(100 * (f + 1) / n_mf)
            title.set_text(f"Phase 1: MPPI Planning — {pct}%")
        else:
            fi = f - n_mf
            k = fly_idx[min(fi, len(fly_idx) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Following MPPI Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
