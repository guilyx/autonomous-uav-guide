# Erwin Lejeune - 2026-02-17
"""flybots — from-scratch algorithms for autonomous UAVs.

Published on PyPI as ``flybots`` and imported as ``flybots``. The import
package was ``uav_sim`` through 1.x and was renamed to match the
distribution in 2.0.0.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flybots")
except PackageNotFoundError:  # pragma: no cover - only when run from a bare checkout
    # Not installed, so there is no metadata to read. Better an honest
    # placeholder than a number hard-coded here that can disagree with
    # pyproject -- a disagreement nothing catches until a release fails.
    __version__ = "0+unknown"
