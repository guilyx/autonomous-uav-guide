# Erwin Lejeune - 2026-02-16
"""Smoke test: verify the package is importable and reports its version."""

from importlib.metadata import version


def test_import():
    import uav_sim

    assert uav_sim.__version__


def test_version_matches_the_installed_distribution():
    """``__version__`` has to be the version that was actually installed.

    It used to be a literal repeated in `pyproject.toml`, here, and in this
    test, so the test compared one hard-coded string to another and could
    not notice them drifting apart. A `v1.0.0` tag against a pyproject
    still reading 0.2.0 is what that drift costs at release time.
    """
    import uav_sim

    assert uav_sim.__version__ == version("flybots")
