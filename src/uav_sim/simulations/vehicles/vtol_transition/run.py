# Erwin Lejeune - 2026-02-19
"""VTOL tilt-rotor transition: hover -> cruise -> hover.

Flown by the library's mode-scheduled VTOL controller, which owns the hard
part: lift authority migrates from the rotors to the wing as the rotors
tilt forward, and one altitude law has to hold through all of it.

The rotors tilt the **full 90°** into wing-borne cruise and back. An
earlier version of this demo hand-rolled a PD loop and ramped the tilt to
30°, which never leaves rotor-borne flight — the aircraft was a slow
quadrotor with a wing along for the ride, and the "transition" the title
promised never happened.

.. code-block:: text

    HOVER ──airspeed builds──▶ TRANSITION ──wing-borne──▶ CRUISE
      ▲                                                     │
      └──────────── BACK_TRANSITION ◀───────decelerate──────┘

Reference: R. Bapst et al., "Design and Implementation of an Unmanned
Tail-Sitter," IROS, 2015.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control.vtol_controller import VTOLCommand, VTOLController, VTOLMode
from uav_sim.logging import SimLogger
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.vehicles.vtol import Tiltrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_vtol_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 700.0
TARGET_ALT = 40.0
CRUISE_AIRSPEED = 22.0
DT = 0.005
DURATION = 68.0
CRUISE_START = 8.0
CRUISE_END = 32.0

_MODE_COLORS = {
    VTOLMode.HOVER: "tab:blue",
    VTOLMode.TRANSITION: "tab:orange",
    VTOLMode.CRUISE: "tab:green",
    VTOLMode.BACK_TRANSITION: "tab:red",
}


def main() -> None:
    vtol = Tiltrotor()
    state = np.zeros(12)
    state[:3] = [30.0, WORLD_SIZE / 2.0, TARGET_ALT]
    vtol.reset(state=state)

    pilot = VTOLController(vtol.vtol_params)

    steps = int(DURATION / DT)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))
    tilt_angles = np.zeros(steps)
    airspeeds = np.zeros(steps)
    wing_fraction = np.zeros(steps)
    modes: list[VTOLMode] = []

    n_steps = steps
    for i in range(steps):
        t = i * DT
        positions[i] = vtol.state[:3]
        eulers[i] = vtol.state[3:6]
        tilt_angles[i] = vtol.tilt
        airspeeds[i] = vtol.airspeed
        weight = vtol.vtol_params.mass * vtol.vtol_params.gravity
        wing_fraction[i] = float(np.clip(vtol.wing_lift / weight, 0.0, 1.5))

        cmd = VTOLCommand(
            altitude=TARGET_ALT,
            cruise=CRUISE_START <= t < CRUISE_END,
            cruise_airspeed=CRUISE_AIRSPEED,
            heading=0.0,
        )
        vtol.step(pilot.compute(vtol.state, vtol.tilt, cmd, DT), DT)
        modes.append(pilot.mode)

        if not np.all(np.isfinite(vtol.state)):
            n_steps = i + 1
            break

    positions = positions[:n_steps]
    eulers = eulers[:n_steps]
    tilt_angles = tilt_angles[:n_steps]
    airspeeds = airspeeds[:n_steps]
    wing_fraction = wing_fraction[:n_steps]
    modes = modes[:n_steps]
    times = np.arange(n_steps) * DT

    cruise_mask = np.array([m is VTOLMode.CRUISE for m in modes])
    alt_err = np.abs(positions[:, 2] - TARGET_ALT)

    logger = SimLogger("vtol_transition", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "VTOL Tilt-Rotor")
    logger.log_metadata("dt", DT)
    logger.log_metadata("duration", float(times[-1]))
    logger.log_metadata("cruise_airspeed_cmd", CRUISE_AIRSPEED)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=positions[i],
            euler=eulers[i],
            tilt_angle=tilt_angles[i],
            airspeed=airspeeds[i],
            wing_lift_fraction=wing_fraction[i],
            mode=modes[i].value,
        )
    logger.log_summary("max_tilt_deg", float(np.degrees(tilt_angles.max())))
    logger.log_summary("final_altitude_m", float(positions[-1, 2]))
    logger.log_summary("max_altitude_error_m", float(alt_err.max()))
    logger.log_summary("reached_cruise", bool(cruise_mask.any()))
    logger.log_summary(
        "cruise_airspeed_ms", float(airspeeds[cruise_mask].mean()) if cruise_mask.any() else 0.0
    )
    logger.log_summary(
        "cruise_wing_lift_fraction",
        float(wing_fraction[cruise_mask].mean()) if cruise_mask.any() else 0.0,
    )
    logger.log_summary("returned_to_hover", bool(modes[-1] is VTOLMode.HOVER))
    logger.save()

    viz = ThreePanelViz(
        title="VTOL Hover → Cruise → Hover",
        world_size=WORLD_SIZE,
        z_max=120.0,
        figsize=(16, 8),
    )

    ax_d = viz.setup_data_axes(ylabel="Tilt [deg] / Airspeed [m/s]", title="Transition Schedule")
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(-5, 118)
    # Shade the mode the aircraft is actually in, so the tilt ramp reads as
    # the consequence of a mode switch rather than an arbitrary schedule.
    boundaries = [0]
    for i in range(1, n_steps):
        if modes[i] is not modes[i - 1]:
            boundaries.append(i)
    boundaries.append(n_steps - 1)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        ax_d.axvspan(times[a], times[b], color=_MODE_COLORS[modes[a]], alpha=0.12, lw=0, zorder=0)
        ax_d.text(
            0.5 * (times[a] + times[b]),
            114,
            modes[a].value.replace("_", " "),
            fontsize=5,
            ha="center",
            va="top",
            color=_MODE_COLORS[modes[a]],
        )
    (tilt_line,) = ax_d.plot([], [], "darkorange", lw=1.0, label="Rotor tilt [deg]")
    (spd_line,) = ax_d.plot([], [], "cyan", lw=1.0, label="Airspeed [m/s]")
    (wing_line,) = ax_d.plot([], [], "limegreen", lw=1.0, label="Wing lift [% weight]")
    ax_d.legend(fontsize=6, loc="center right")

    anim = SimAnimator("vtol_transition", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="dodgerblue")

    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    vehicle_arts_3d: list = []
    title = viz.ax3d.set_title("VTOL Transition")

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, positions, k)

        clear_vehicle_artists(vehicle_arts_3d)
        R = Quadrotor.rotation_matrix(*eulers[k])
        # Draw the tilt-rotor at its actual rotor angle. The transition is
        # the subject of this simulation, so illustrating it with a
        # fixed-geometry quadrotor hid the one thing worth watching.
        vehicle_arts_3d.extend(
            draw_vtol_3d(
                viz.ax3d,
                positions[k],
                R,
                tilt=float(tilt_angles[k]),
                fuselage_length=30.0,
                wingspan=48.0,
                arm_length=10.0,
                body_color=_MODE_COLORS[modes[k]],
                wing_color=_MODE_COLORS[modes[k]],
                tail_color=_MODE_COLORS[modes[k]],
            )
        )

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dot_top,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=5)
        viz._vehicle_arts_top.append(dot_top)

        tilt_line.set_data(times[:k], np.degrees(tilt_angles[:k]))
        spd_line.set_data(times[:k], airspeeds[:k])
        wing_line.set_data(times[:k], 100.0 * wing_fraction[:k])
        title.set_text(
            f"VTOL — t={times[k]:.1f}s  {modes[k].value.replace('_', ' ')}  "
            f"tilt={np.degrees(tilt_angles[k]):.0f}°  Va={airspeeds[k]:.1f}m/s  "
            f"alt={positions[k, 2]:.1f}m"
        )

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
