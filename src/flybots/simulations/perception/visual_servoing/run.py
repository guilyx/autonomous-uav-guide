"""Legacy visual-servoing entrypoint.

Use dedicated variants instead:
  - perception/visual_servoing_gimbal
  - perception/visual_servoing_fixed
"""

from __future__ import annotations

from pathlib import Path

from flybots.simulations.perception.visual_servoing._core import run_visual_servoing


def main() -> None:
    run_visual_servoing(
        sim_name="visual_servoing",
        out_dir=Path(__file__).parent,
        gimbal_tracking=True,
    )


if __name__ == "__main__":
    main()
