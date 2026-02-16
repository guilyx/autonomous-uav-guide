# Erwin Lejeune - 2026-02-16
"""Smoke test: verify the package is importable."""


def test_import():
    import quadrotor_sim

    assert quadrotor_sim.__version__ == "0.1.0"
