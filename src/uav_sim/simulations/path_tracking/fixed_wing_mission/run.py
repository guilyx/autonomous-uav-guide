# Erwin Lejeune - 2026-08-18
"""Fixed-wing mission navigation: waypoints, racetrack and return-to-launch.

One flight, three mission modes, flown by the same guidance layer over the
same successive-loop-closure autopilot:

1. **Waypoint legs** — four points and a climb, followed with a cross-track
   vector field rather than by pointing the nose at the next waypoint.
2. **Racetrack** — two straight legs joined by half-orbits, flown twice.
3. **Return-to-launch** — triggered mid-pattern: climb to a safe altitude,
   transit home, and loiter.

The point of putting all three in one clip is the hand-overs. Each mode
change happens while the aircraft is somewhere the plan did not anticipate,
and the guidance has to pick it up from there — which is exactly what an
RTL has to do for real.

The airframe is the 0.6 kg trainer, chosen so the whole mission fits a
500 m world: it cruises at 12 m/s and its bank-limited turn radius is under
15 m, where the Aerosonde would need 125 m and a world eight times the
size.

Reference: R. W. Beard, T. W. McLain, "Small Unmanned Aircraft: Theory and
Practice," Princeton University Press, 2012, Chapters 10-11.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

from uav_sim.control.fixed_wing_autopilot import FixedWingAutopilot
from uav_sim.guidance import (
    FixedWingMission,
    GuidanceGains,
    LineLeg,
    MissionLeg,
    OrbitLeg,
    racetrack_plan,
    waypoint_plan,
)
from uav_sim.logging import SimLogger
from uav_sim.vehicles.fixed_wing import FixedWingPreset, create_fixed_wing
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_fixed_wing_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 500.0
Z_MAX = 180.0
DT = 0.005
MAX_DURATION = 330.0

CRUISE_SPEED = 12.0
HOME = np.array([60.0, 60.0, 70.0])
WAYPOINTS = [
    HOME,
    np.array([420.0, 90.0, 90.0]),
    np.array([430.0, 330.0, 110.0]),
    np.array([150.0, 380.0, 110.0]),
]
RACETRACK_CENTRE = np.array([250.0, 240.0, 110.0])
RACETRACK_LENGTH = 200.0
RACETRACK_RADIUS = 45.0
RACETRACK_HEADING = np.radians(20.0)
RACETRACK_LAPS = 2

RTL_SAFE_ALTITUDE = 145.0
RTL_LOITER_RADIUS = 60.0
RTL_HOLD_TIME = 70.0
"""Seconds of loiter to record once the aircraft is home."""

_PHASE_COLOURS = {"waypoints": "deepskyblue", "racetrack": "goldenrod", "rtl": "orangered"}
_SETTLE_TIME = 30.0
"""Seconds after a mode change before the phase counts as settled."""
_ERROR_FLOOR = 1.0
"""Floor [m] on the plotted error.

About the trainer's wingspan, and below the point where a distinction is
worth drawing. Without it every zero crossing dives to the bottom of a log
axis and the plot reads as noise rather than as tracking.
"""


def leg_polyline(
    leg: MissionLeg,
    previous: MissionLeg | None = None,
    samples: int = 96,
) -> NDArray[np.floating]:
    """Sample a mission leg into a polyline for plotting.

    A partial orbit is drawn starting from where the previous straight
    handed over, so the arc on the plot is the same half of the circle the
    aircraft actually flies rather than an arbitrary one.
    """
    if isinstance(leg, LineLeg):
        return np.linspace(leg.line.origin, leg.target, samples)
    if not isinstance(leg, OrbitLeg):
        raise TypeError(f"no polyline is defined for a {type(leg).__name__}")
    orbit = leg.orbit
    sweep = leg.sweep if leg.sweep is not None else 2.0 * np.pi
    start = orbit.angle_at(previous.target) if isinstance(previous, LineLeg) else 0.0
    angles = start + int(orbit.direction) * np.linspace(0.0, sweep, samples)
    return np.stack(
        [
            orbit.centre[0] + orbit.radius * np.cos(angles),
            orbit.centre[1] + orbit.radius * np.sin(angles),
            np.full(samples, orbit.altitude),
        ],
        axis=1,
    )


def main() -> None:
    aircraft = create_fixed_wing(FixedWingPreset.MINI_TRAINER)
    aircraft.reset_trimmed(airspeed=CRUISE_SPEED, altitude=float(HOME[2]), heading=0.0)
    # `state` hands back a copy, so the start position has to go through
    # reset() rather than being poked into the array it returns.
    start_state = aircraft.state
    start_state[0], start_state[1] = HOME[0], HOME[1]
    aircraft.reset(state=start_state)

    pilot = FixedWingAutopilot(aircraft.fw_params)
    gains = GuidanceGains(gravity=aircraft.fw_params.gravity)
    turn_radius = gains.turn_radius(CRUISE_SPEED)

    survey = waypoint_plan(WAYPOINTS, airspeed=CRUISE_SPEED, gains=gains)
    circuit = racetrack_plan(
        RACETRACK_CENTRE,
        length=RACETRACK_LENGTH,
        radius=RACETRACK_RADIUS,
        heading=RACETRACK_HEADING,
        airspeed=CRUISE_SPEED,
        gains=gains,
    )
    mission = FixedWingMission(aircraft.fw_params, survey, gains, home=HOME)
    drawn: list[tuple[str, MissionLeg]] = [("waypoints", leg) for leg in survey]

    steps = int(MAX_DURATION / DT)
    positions = np.zeros((steps, 3))
    eulers = np.zeros((steps, 3))
    airspeeds = np.zeros(steps)
    path_errors = np.zeros(steps)
    altitude_cmds = np.zeros(steps)
    bank_ff = np.zeros(steps)
    phases: list[str] = []
    labels: list[str] = []

    phase = "waypoints"
    rtl_started: float | None = None
    used = steps

    for i in range(steps):
        t = i * DT
        state = aircraft.state
        positions[i] = state[:3]
        eulers[i] = state[3:6]
        airspeeds[i] = float(np.linalg.norm(state[6:9]))

        # ── mission mode scheduling ───────────────────────────────────────
        if phase == "waypoints" and mission.is_complete:
            mission.fly(circuit, loop=True)
            drawn.extend(("racetrack", leg) for leg in circuit)
            phase = "racetrack"
        elif phase == "racetrack" and mission.diagnostics.laps >= RACETRACK_LAPS:
            mission.return_to_launch(
                state,
                safe_altitude=RTL_SAFE_ALTITUDE,
                airspeed=CRUISE_SPEED,
                loiter_radius=RTL_LOITER_RADIUS,
            )
            drawn.extend(("rtl", leg) for leg in mission.legs)
            phase = "rtl"
            rtl_started = t
        elif phase == "rtl" and rtl_started is not None and t - rtl_started > RTL_HOLD_TIME:
            used = i
            break

        command = mission.update(state)
        path_errors[i] = mission.diagnostics.path_error
        altitude_cmds[i] = command.altitude
        bank_ff[i] = command.roll_feedforward
        phases.append(phase)
        labels.append(mission.diagnostics.leg_label)

        aircraft.step(pilot.compute(state, command, DT), DT)
        if not np.all(np.isfinite(aircraft.state)):
            used = i + 1
            break

    positions = positions[:used]
    eulers = eulers[:used]
    airspeeds = airspeeds[:used]
    path_errors = path_errors[:used]
    altitude_cmds = altitude_cmds[:used]
    bank_ff = bank_ff[:used]
    phases = phases[:used]
    labels = labels[:used]
    times = np.arange(used) * DT

    phase_starts = {name: float(times[phases.index(name)]) for name in dict.fromkeys(phases)}

    # ── logging ───────────────────────────────────────────────────────────
    logger = SimLogger("fixed_wing_mission", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "Fixed-Wing Mission Navigation")
    logger.log_metadata("airframe", FixedWingPreset.MINI_TRAINER.value)
    logger.log_metadata("dt", DT)
    logger.log_metadata("duration", float(times[-1]))
    logger.log_metadata("cruise_airspeed", CRUISE_SPEED)
    logger.log_metadata("turn_radius_m", float(turn_radius))
    logger.log_metadata("transition_distance_m", float(gains.transition_distance(CRUISE_SPEED)))
    for i in range(used):
        logger.log_step(
            t=times[i],
            position=positions[i],
            euler=eulers[i],
            airspeed=airspeeds[i],
            path_error=path_errors[i],
            altitude_command=altitude_cmds[i],
            phase=phases[i],
            leg=labels[i],
        )
    for name, start in phase_starts.items():
        logger.log_summary(f"{name}_start_s", start)
    # Each phase is scored twice, because the two numbers answer different
    # questions. The capture peak is how far off the aircraft was when the
    # mode changed under it — large by construction, and the thing the
    # guidance exists to remove. The settled figures are how well it holds
    # the path once it is on it.
    for name in phase_starts:
        mask = np.array([p == name for p in phases])
        capture = mask & (times <= phase_starts[name] + _SETTLE_TIME)
        settled = mask & (times > phase_starts[name] + _SETTLE_TIME)
        if capture.any():
            logger.log_summary(
                f"{name}_capture_peak_path_error_m", float(np.abs(path_errors[capture]).max())
            )
        if settled.any():
            logger.log_summary(
                f"{name}_settled_mean_path_error_m", float(np.abs(path_errors[settled]).mean())
            )
            logger.log_summary(
                f"{name}_settled_max_path_error_m", float(np.abs(path_errors[settled]).max())
            )
    home_distance = float(np.linalg.norm(positions[-1, :2] - HOME[:2]))
    logger.log_summary("final_distance_from_home_m", home_distance)
    logger.log_summary("final_altitude_m", float(positions[-1, 2]))
    logger.log_summary("min_airspeed_ms", float(airspeeds.min()))
    logger.log_summary("stall_speed_ms", float(aircraft.fw_params.stall_airspeed))
    logger.save()

    # ── visualisation ─────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Fixed-Wing Mission Navigation — Waypoints, Racetrack, Return-to-Launch",
        world_size=WORLD_SIZE,
        z_max=Z_MAX,
        figsize=(16, 8),
    )

    seen: set[str] = set()
    for index, (name, leg) in enumerate(drawn):
        previous = drawn[index - 1][1] if index and drawn[index - 1][0] == name else None
        polyline = leg_polyline(leg, previous)
        label = name.replace("rtl", "return to launch") if name not in seen else ""
        seen.add(name)
        viz.draw_path(polyline, color=_PHASE_COLOURS[name], lw=1.1, alpha=0.55, label=label)

    waypoint_xy = np.array([w[:2] for w in WAYPOINTS])
    viz.ax_top.plot(waypoint_xy[:, 0], waypoint_xy[:, 1], "wo", ms=4, mec="black", zorder=6)
    viz.ax_top.plot(HOME[0], HOME[1], "g^", ms=9, zorder=7)
    viz.ax3d.scatter(*HOME, c="lime", s=90, marker="^", zorder=6, label="Home")
    viz.ax3d.legend(fontsize=7, loc="upper left")

    ax_err = viz.setup_data_axes(
        ylabel="|Path error| [m]", title="Distance from the commanded path"
    )
    ax_err.set_xlim(0, times[-1])
    # Log magnitude rather than signed error on a linear axis. Each mode
    # change throws the error two orders of magnitude above where it
    # settles, so a linear axis wide enough to show a capture flattens
    # everything in between onto the zero line.
    ax_err.set_yscale("log")
    error_magnitude = np.maximum(np.abs(path_errors), _ERROR_FLOOR)
    ax_err.set_ylim(_ERROR_FLOOR, max(20.0, float(error_magnitude.max()) * 1.5))
    ax_err.axhline(turn_radius, color="grey", lw=0.6, ls=":", alpha=0.6)
    for name, start in phase_starts.items():
        ax_err.axvline(start, color=_PHASE_COLOURS[name], lw=0.8, ls="--", alpha=0.7)
    (err_line,) = ax_err.plot([], [], color="deepskyblue", lw=1.0, label="|Path error|")

    ax_alt = ax_err.twinx()
    ax_alt.set_ylabel("Altitude [m]", fontsize=7)
    ax_alt.tick_params(labelsize=6)
    ax_alt.set_ylim(0, Z_MAX)
    ax_alt.plot(times, altitude_cmds, color="orange", lw=0.6, ls="--", alpha=0.6, label="Alt cmd")
    (alt_line,) = ax_alt.plot([], [], color="orange", lw=1.0, label="Altitude")
    ax_err.legend(fontsize=5, loc="upper left")
    ax_alt.legend(fontsize=5, loc="lower right")

    # This mission runs four times longer than a typical demo, so the
    # frame budget and resolution are trimmed to keep the rendered GIF in
    # line with the rest of the catalogue rather than four times their size.
    anim = SimAnimator("fixed_wing_mission", out_dir=Path(__file__).parent, dpi=64)
    anim._fig = viz.fig

    trail = viz.create_trail_artists(color="black", lw=1.0)
    skip = max(1, used // 130)
    frames = list(range(0, used, skip))
    vehicle_3d: list = []
    title = viz.ax3d.set_title("Fixed-Wing Mission")

    def update(f: int) -> None:
        k = frames[min(f, len(frames) - 1)]
        viz.update_trail(trail, positions, k)

        clear_vehicle_artists(vehicle_3d)
        rotation = Quadrotor.rotation_matrix(*eulers[k])
        vehicle_3d.extend(draw_fixed_wing_3d(viz.ax3d, positions[k], rotation, scale=28.0))

        clear_vehicle_artists(viz._vehicle_arts_top)
        (dot,) = viz.ax_top.plot(positions[k, 0], positions[k, 1], "ko", ms=5, zorder=8)
        viz._vehicle_arts_top.append(dot)

        err_line.set_data(times[:k], error_magnitude[:k])
        alt_line.set_data(times[:k], positions[:k, 2])
        title.set_text(
            f"{phases[k]} — {labels[k]}   t={times[k]:.0f}s  "
            f"err={path_errors[k]:+.1f}m  alt={positions[k, 2]:.0f}m  "
            f"φff={np.degrees(bank_ff[k]):+.0f}°"
        )

    anim.animate(update, len(frames))
    anim.save()


if __name__ == "__main__":
    main()
