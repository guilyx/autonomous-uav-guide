# Erwin Lejeune - 2026-02-19
"""NMPC online trajectory tracking through an urban environment.

The drone takes off, then a nonlinear MPC (single-shooting, RK2
integration) runs as a receding-horizon controller at 50 Hz. A
pure-pursuit carrot on a pre-planned global path provides the
reference position and velocity for NMPC at each step.

Reference: M. Diehl et al., "Real-Time Optimization and Nonlinear Model
Predictive Control of Processes Governed by Differential-Algebraic
Equations," J. Process Control, 2002. DOI: 10.1016/S0959-1524(02)00023-1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_tracking.nmpc import NMPCTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0
DT_SIM = 0.005
DT_CTRL = 0.02  # 50 Hz


def main() -> None:
    world, buildings = default_world()

    global_path = np.array(
        [
            [3.0, 3.0, CRUISE_ALT],
            [8.0, 6.0, CRUISE_ALT],
            [15.0, 10.0, CRUISE_ALT],
            [20.0, 18.0, CRUISE_ALT],
            [24.0, 24.0, CRUISE_ALT],
            [27.0, 27.0, CRUISE_ALT],
        ]
    )

    quad = Quadrotor()
    quad.reset(position=np.array([global_path[0, 0], global_path[0, 1], 0.0]))
    ctrl = CascadedPIDController()

    states_list: list[np.ndarray] = []
    local_goals: list[np.ndarray] = []

    # Takeoff
    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=DT_SIM, duration=3.0, states=states_list)
    init_hover(quad)

    nmpc = NMPCTracker(
        horizon=8,
        dt=DT_CTRL,
        mass=quad.params.mass,
        gravity=quad.params.gravity,
        inertia=quad.params.inertia,
    )
    pursuit = PurePursuit3D(lookahead=4.0, waypoint_threshold=2.0, adaptive=True)

    sim_steps_per_ctrl = max(1, int(DT_CTRL / DT_SIM))
    max_ctrl_steps = 2000
    wrench = quad.hover_wrench()

    for _ in range(max_ctrl_steps):
        s = quad.state
        if not (np.all(np.isfinite(s[:3])) and np.all(np.abs(s[:3]) < 500)):
            break

        vel = s[6:9] if len(s) >= 9 else None
        target = pursuit.compute_target(s[:3], global_path, velocity=vel)
        local_goals.append(target.copy())

        direction = target - s[:3]
        dist = float(np.linalg.norm(direction))
        ref_vel = direction / max(dist, 0.01) * 2.0 if dist > 0.3 else np.zeros(3)

        wrench = nmpc.compute(s, target, ref_vel=ref_vel)

        for _ in range(sim_steps_per_ctrl):
            states_list.append(quad.state.copy())
            quad.step(wrench, DT_SIM)

        if pursuit.is_path_complete(s[:3], global_path):
            break

    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    flight_pos = flight_states[:, :3]
    n_total = len(flight_pos)

    # ── Animation ─────────────────────────────────────────────────────
    skip = max(1, n_total // 200)
    frames = list(range(0, n_total, skip))
    n_frames = len(frames)

    viz = ThreePanelViz(title="NMPC — Online Trajectory Tracking", world_size=WORLD_SIZE)
    viz.draw_buildings(buildings)
    viz.draw_path(global_path, color="cyan", lw=1.5, alpha=0.4, label="Global Path")
    viz.mark_start_goal(global_path[0], global_path[-1])

    trail = viz.create_trail_artists(color="orange")
    (lg_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10, label="Local Goal")
    (lg_top,) = viz.ax_top.plot([], [], "r*", ms=8, zorder=10)
    viz.ax3d.legend(fontsize=7, loc="upper left")

    anim = SimAnimator("nmpc", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    local_goal_arr = np.array(local_goals) if local_goals else np.zeros((1, 3))

    def update(f: int) -> None:
        k = frames[f]
        viz.update_trail(trail, flight_pos, k)
        viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)

        lg_idx = min(k // sim_steps_per_ctrl, len(local_goal_arr) - 1)
        lg = local_goal_arr[lg_idx]
        lg_3d.set_data([lg[0]], [lg[1]])
        lg_3d.set_3d_properties([lg[2]])
        lg_top.set_data([lg[0]], [lg[1]])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
