# Erwin Lejeune - 2026-08-27
"""Geofence and floor: a controller commanded straight through both.

The nominal controller flies a racetrack that is deliberately too big for
the box and dips below the floor on every lap. It is not trying to escape
once; it is trying to escape continuously, which is what makes the fence
visible: the commanded oval is repeatedly carved back into the safe volume,
lap after lap.

The first version of this simulation flew flat out at one wall, stopped,
then dived at the ground and stopped. Both barriers held, and the result
was two straight lines and two dead stops -- nothing that showed *how* a
high-order barrier behaves. Both constraints are on position, which the
input reaches only through the second derivative, so the filter must begin
decelerating well before the limit rather than clamping at it. That is far
easier to see against a curve than against a halt.

A clamp is not a safety guarantee. It is a report that safety was already
lost, and the unfiltered run shows what that looks like.

Reference: W. Xiao & C. Belta, "High Order Control Barrier Functions,"
IEEE TAC, 2022. DOI: 10.1109/TAC.2021.3105491
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from flybots.logging import SimLogger
from flybots.safety import (
    AltitudeFloorBarrier,
    GeofenceBoxBarrier,
    SafetyFilter,
    SpeedLimitBarrier,
)
from flybots.visualization import SimAnimator, attitude_from_velocity
from flybots.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

LOWER = np.array([5.0, 5.0, 0.0])
UPPER = np.array([55.0, 55.0, 45.0])
FLOOR = 8.0
MAX_SPEED = 9.0
MAX_ACCEL = 10.0
DT = 0.02
STEPS = 3400
OMEGA = 0.55
# Deliberately oversized: the commanded oval runs well outside the box in x
# and y, and its vertical swing takes it under the floor.
CMD_RX = 42.0
CMD_RY = 34.0
CMD_DZ = 22.0


def _fly(use_filter: bool):
    filt = SafetyFilter(
        [
            GeofenceBoxBarrier(LOWER, UPPER, k1=3.5, k2=3.5),
            AltitudeFloorBarrier(FLOOR, k1=3.5, k2=3.5),
            SpeedLimitBarrier(MAX_SPEED),
        ],
        u_min=-MAX_ACCEL,
        u_max=MAX_ACCEL,
    )

    centre = np.array([30.0, 30.0, 24.0])
    # Start at the centre, inside the box. Starting on the commanded oval
    # puts the vehicle at x = 72 against a wall at x = 55 -- outside the safe
    # set before the first step, which a barrier cannot undo and which then
    # reports as a 17 m violation the filter never actually caused.
    pos = np.array([centre.copy()])
    vel = np.zeros_like(pos)
    history = np.zeros((STEPS, 3))
    wall_margin = np.zeros(STEPS)
    floor_margin = np.zeros(STEPS)
    commanded = np.zeros((STEPS, 3))

    for step in range(STEPS):
        t = step * DT
        # An oval that does not fit, tracked by a plain PD. Nothing in the
        # nominal controller knows the box exists.
        target = centre + np.array(
            [
                CMD_RX * np.cos(OMEGA * t),
                CMD_RY * np.sin(OMEGA * t),
                CMD_DZ * np.sin(2.0 * OMEGA * t),
            ]
        )
        commanded[step] = target
        nominal = np.clip(3.0 * (target - pos) - 3.2 * vel, -MAX_ACCEL, MAX_ACCEL)
        command = filt(pos, vel, nominal).command if use_filter else nominal
        vel = vel + command * DT
        pos = pos + vel * DT

        history[step] = pos[0]
        wall_margin[step] = float(min(np.min(pos[0] - LOWER), np.min(UPPER - pos[0])))
        floor_margin[step] = float(pos[0, 2] - FLOOR)

    return history, wall_margin, floor_margin, commanded


def main() -> None:
    safe, wall_on, floor_on, commanded = _fly(True)
    loose, wall_off, floor_off, _ = _fly(False)
    times = np.arange(STEPS) * DT

    logger = SimLogger("geofence_floor", out_dir=Path(__file__).parent, downsample=10)
    logger.log_metadata("algorithm", "High-order CBF geofence and altitude floor")
    logger.log_metadata("floor_m", FLOOR)
    logger.log_metadata("box_lower", LOWER)
    logger.log_metadata("box_upper", UPPER)
    for step in range(STEPS):
        logger.log_step(
            t=times[step],
            position=safe[step],
            wall_margin=float(wall_on[step]),
            floor_margin=float(floor_on[step]),
            wall_margin_unfiltered=float(wall_off[step]),
            floor_margin_unfiltered=float(floor_off[step]),
        )
    logger.log_summary("min_wall_margin_m", float(wall_on.min()))
    logger.log_summary("min_floor_margin_m", float(floor_on.min()))
    logger.log_summary("min_wall_margin_unfiltered_m", float(wall_off.min()))
    logger.log_summary("min_floor_margin_unfiltered_m", float(floor_off.min()))
    logger.log_summary("contained", bool(wall_on.min() > -0.5 and floor_on.min() > -0.5))
    logger.save()

    skip = max(1, STEPS // 140)
    idx = list(range(0, STEPS, skip))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_wall = fig.add_subplot(gs[0, 1])
    ax_floor = fig.add_subplot(gs[1, 1])

    fig.suptitle("Geofence and Floor — an oval that does not fit, carved to size", fontsize=13)

    ax3d.set_xlim(0, 60)
    ax3d.set_ylim(0, 60)
    ax3d.set_zlim(0, 50)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    # The fence, drawn as its footprint and its floor.
    bx = [LOWER[0], UPPER[0], UPPER[0], LOWER[0], LOWER[0]]
    by = [LOWER[1], LOWER[1], UPPER[1], UPPER[1], LOWER[1]]
    for z in (FLOOR, UPPER[2]):
        ax3d.plot(bx, by, [z] * 5, color="crimson", lw=1.2, alpha=0.8)
    # The oval that was asked for, against the path that was allowed.
    ax3d.plot(
        commanded[:, 0],
        commanded[:, 1],
        commanded[:, 2],
        color="0.6",
        lw=1.0,
        ls="--",
        label="commanded",
    )
    ax3d.plot(loose[:, 0], loose[:, 1], loose[:, 2], color="tab:red", lw=1.0, alpha=0.5)

    ax_wall.set_xlim(0, times[-1])
    ax_wall.set_ylim(-20, 30)
    ax_wall.set_xlabel("Time [s]", fontsize=8)
    ax_wall.set_ylabel("Margin to nearest wall [m]", fontsize=8)
    ax_wall.set_title("Geofence — negative is outside", fontsize=9)
    ax_wall.grid(True, alpha=0.3)
    ax_wall.axhline(0.0, color="k", lw=1.0)
    (w_on,) = ax_wall.plot([], [], color="tab:green", lw=1.5, label="filtered")
    (w_off,) = ax_wall.plot([], [], color="tab:red", lw=1.2, label="unfiltered")
    ax_wall.legend(fontsize=7, loc="lower left")

    ax_floor.set_xlim(0, times[-1])
    ax_floor.set_ylim(-25, 30)
    ax_floor.set_xlabel("Time [s]", fontsize=8)
    ax_floor.set_ylabel("Height above floor [m]", fontsize=8)
    ax_floor.set_title("Altitude floor — negative is underground", fontsize=9)
    ax_floor.grid(True, alpha=0.3)
    ax_floor.axhline(0.0, color="k", lw=1.0)
    (f_on,) = ax_floor.plot([], [], color="tab:green", lw=1.5, label="filtered")
    (f_off,) = ax_floor.plot([], [], color="tab:red", lw=1.2, label="unfiltered")
    ax_floor.legend(fontsize=7, loc="lower left")

    (trail,) = ax3d.plot([], [], [], color="tab:green", lw=1.8)
    veh: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("geofence_floor", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(frame: int) -> None:
        step = idx[frame]
        trail.set_data(safe[: step + 1, 0], safe[: step + 1, 1])
        trail.set_3d_properties(safe[: step + 1, 2])

        clear_vehicle_artists(veh)
        v = (safe[step] - safe[step - 1]) / DT if step > 0 else np.zeros(3)
        veh.extend(draw_quadrotor_3d(ax3d, safe[step], attitude_from_velocity(v), size=1.6))

        w_on.set_data(times[:step], wall_on[:step])
        w_off.set_data(times[:step], wall_off[:step])
        f_on.set_data(times[:step], floor_on[:step])
        f_off.set_data(times[:step], floor_off[:step])
        title.set_text(
            f"t = {step * DT:.1f} s   wall {wall_on[step]:+.1f} m   floor {floor_on[step]:+.1f} m"
        )

    anim.animate(update, len(idx))
    anim.save()

    print(f"  filtered   wall {wall_on.min():+.2f} m   floor {floor_on.min():+.2f} m")
    print(f"  unfiltered wall {wall_off.min():+.2f} m   floor {floor_off.min():+.2f} m")


if __name__ == "__main__":
    main()
