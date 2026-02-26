"""Visual servoing with fixed camera (drone-only control)."""

from __future__ import annotations

from pathlib import Path

from uav_sim.simulations.perception.visual_servoing._core import run_visual_servoing


def main() -> None:
    run_visual_servoing(
        sim_name="visual_servoing_fixed",
        out_dir=Path(__file__).parent,
        gimbal_tracking=False,
    )


if __name__ == "__main__":
    main()
