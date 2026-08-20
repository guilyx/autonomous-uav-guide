# Erwin Lejeune - 2026-02-17
"""flybots — from-scratch algorithms for autonomous UAVs.

Published on PyPI as ``flybots``; the import package is still ``uav_sim``
and will be renamed to match in a later release.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flybots")
except PackageNotFoundError:  # pragma: no cover - only when run from a bare checkout
    # Not installed, so there is no metadata to read. Better an honest
    # placeholder than a number hard-coded here that can disagree with
    # pyproject -- a disagreement nothing catches until a release fails.
    __version__ = "0+unknown"
