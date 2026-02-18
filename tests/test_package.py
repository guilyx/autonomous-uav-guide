# Erwin Lejeune - 2026-02-16
"""Smoke test: verify the package is importable."""


def test_import():
    import uav_sim

    assert uav_sim.__version__ == "0.2.0"
