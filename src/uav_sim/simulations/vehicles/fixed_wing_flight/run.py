# Erwin Lejeune - 2026-02-19
"""Fixed-wing level flight and gentle banked turn with closed-loop autopilot.

The aircraft starts from a solved **trim** condition and is flown by the
library's successive-loop-closure autopilot: airspeed on throttle,
altitude through a pitch command, course through a bank command, with a
yaw damper on the rudder.  It holds a straight leg, turns 90 degrees while
climbing, then levels out on the new course.

Two things this demo exists to get right, because both were wrong before:

* The airframe has to match the world.  A 13.5 kg Aerosonde trims around
  35 m/s and needs hundreds of metres to turn; asked to hold 8 m/s in a
  30 m box it is below stall speed from the first frame and simply falls
  out of the sky.  The 0.6 kg trainer used here stalls at 6.3 m/s and
  cruises at 12, so a 200 m world is several turn diameters across.
* Starting from trim, not from a guess.  An untrimmed start spends the
  first seconds porpoising, which is the transient — not the controller —
  that dominates a short clip.

Reference: R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and
Practice," Princeton University Press, 2012, Chapters 5 and 6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control.fixed_wing_autopilot import AutopilotCommand, FixedWingAutopilot
from uav_sim.logging import SimLogger
from uav_sim.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_fixed_wing_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 200.0
CRUISE_ALT = 40.0
CLIMB_ALT = 65.0
DESCEND_ALT = 45.0
CRUISE_SPEED = 12.0
START_XY = (45.0, 20.0)
DT = 0.005
DURATION = 62.0

# Rounded-square circuit: a straight leg, a climbing turn, a cruise leg,
# then a descending turn back onto the original course. Leg lengths are
# sized from cruise speed so the whole circuit fits the world.
_LEGS = (
    (11.0, 0.0, CRUISE_ALT),
    (24.0, np.pi / 2, CLIMB_ALT),
    (37.0, np.pi, CLIMB_ALT),
    (50.0, -np.pi / 2, DESCEND_ALT),
)


def _command(t: float) -> AutopilotCommand:
    """Mission schedule: straight leg → climbing turn → cruise → descending turn."""
    for end_t, course, altitude in _LEGS:
        if t < end_t:
            return AutopilotCommand(altitude=altitude, airspeed=CRUISE_SPEED, course=float(course))
    return AutopilotCommand(altitude=DESCEND_ALT, airspeed=CRUISE_SPEED, course=0.0)


def main() -> None:
    aircraft = create_fixed_wing(FixedWingPreset.MINI_TRAINER)
    aircraft.reset_trimmed(airspeed=CRUISE_SPEED, altitude=CRUISE_ALT, heading=0.0)
    # `state` hands back a copy, so the start position has to go through
    # reset() rather than being poked into the array it returns.
    start_state = aircraft.state
    start_state[0], start_state[1] = START_XY
    aircraft.reset(state=start_state)

    pilot = FixedWingAutopilot(aircraft.fw_params)

    steps = int(DURATION / DT)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))
    airspeeds = np.zeros(steps)
    alt_cmd = np.zeros(steps)
    roll_cmd = np.zeros(steps)

    n_steps = steps
    for i in range(steps):
        t = i * DT
        positions[i] = aircraft.state[:3]
        eulers[i] = aircraft.state[3:6]
        airspeeds[i] = float(np.linalg.norm(aircraft.state[6:9]))

        cmd = _command(t)
        alt_cmd[i] = cmd.altitude
        control = pilot.compute(aircraft.state, cmd, DT)
        roll_cmd[i] = pilot.diagnostics.roll_cmd
        aircraft.step(control, DT)

        if not np.all(np.isfinite(aircraft.state[:3])):
            n_steps = i + 1
            break

    positions = positions[:n_steps]
    eulers = eulers[:n_steps]
    airspeeds = airspeeds[:n_steps]
    alt_cmd = alt_cmd[:n_steps]
    roll_cmd = roll_cmd[:n_steps]
    times = np.arange(n_steps) * DT

    alt_err = np.abs(positions[:, 2] - alt_cmd)
    # Skip the commanded step changes when scoring: the interesting number
    # is how well it holds, not how fast it slews.
    settled = times > 45.0

    logger = SimLogger("fixed_wing_flight", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Fixed Wing Autopilot")
    logger.log_metadata("airframe", FixedWingPreset.MINI_TRAINER.value)
    logger.log_metadata("dt", DT)
    logger.log_metadata("duration", float(times[-1]))
    logger.log_metadata("cruise_airspeed", CRUISE_SPEED)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=positions[i],
            euler=eulers[i],
            airspeed=airspeeds[i],
            altitude_command=alt_cmd[i],
            roll_command=roll_cmd[i],
        )
    logger.log_summary("final_altitude_m", float(positions[-1, 2]))
    logger.log_summary("settled_altitude_error_m", float(alt_err[settled].mean()))
    airspeed_err = np.abs(airspeeds[settled] - CRUISE_SPEED)
    logger.log_summary("settled_airspeed_error_ms", float(airspeed_err.mean()))
    logger.log_summary("min_airspeed_ms", float(airspeeds.min()))
    logger.log_summary("stall_speed_ms", float(aircraft.fw_params.stall_airspeed))
    logger.save()

    viz = ThreePanelViz(
        title="Fixed-Wing Flight — Trim + Successive-Loop-Closure Autopilot",
        world_size=WORLD_SIZE,
        z_max=100.0,
        figsize=(16, 8),
    )
    ax_d = viz.setup_data_axes(ylabel="Altitude [m] / Airspeed [m/s]", title="Autopilot Tracking")
    ax_d.set_xlim(0, times[-1])
    ax_d.set_ylim(0, max(CLIMB_ALT * 1.4, float(airspeeds.max()) * 1.4))
    ax_d.plot(times, alt_cmd, color="cyan", lw=0.6, ls="--", alpha=0.6, label="Altitude cmd")
    ax_d.axhline(CRUISE_SPEED, color="orange", lw=0.6, ls="--", alpha=0.6, label="Airspeed cmd")
    ax_d.axhline(
        aircraft.fw_params.stall_airspeed, color="red", lw=0.6, ls=":", alpha=0.6, label="Stall"
    )
    (alt_line,) = ax_d.plot([], [], "cyan", lw=1.0, label="Altitude")
    (spd_line,) = ax_d.plot([], [], "orange", lw=1.0, label="Airspeed")
    ax_d.legend(fontsize=6, loc="lower right", ncol=2)

    anim = SimAnimator("fixed_wing_flight", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="royalblue")

    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    vehicle_arts_3d: list = []
    title = viz.ax3d.set_title("Fixed-Wing Flight")

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, positions, k)

        clear_vehicle_artists(vehicle_arts_3d)
        R = Quadrotor.rotation_matrix(*eulers[k])
        vehicle_arts_3d.extend(draw_fixed_wing_3d(viz.ax3d, positions[k], R, scale=22.0))

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dot_top,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=5)
        viz._vehicle_arts_top.append(dot_top)

        alt_line.set_data(times[:k], positions[:k, 2])
        spd_line.set_data(times[:k], airspeeds[:k])
        title.set_text(
            f"Fixed-Wing — t={times[k]:.1f}s  alt={positions[k, 2]:.1f}m  "
            f"Va={airspeeds[k]:.1f}m/s  φ={np.degrees(eulers[k, 0]):+.0f}°"
        )

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
