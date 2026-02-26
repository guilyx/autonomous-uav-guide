"""Reusable simulation setup helpers for quick new runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.logging import SimLogger
from uav_sim.simulations.api import (
    create_environment,
    create_mission,
    create_sim,
    run_sim,
    spawn_quad_platform,
)
from uav_sim.simulations.common import CRUISE_ALT, figure_8_path
from uav_sim.simulations.contracts import RunResult
from uav_sim.simulations.standards import SimulationStandard


def select_standard(profile: str = "flight_coupled") -> SimulationStandard:
    """Return a standard profile by name."""
    if profile == "flight_coupled":
        return SimulationStandard.flight_coupled()
    if profile == "estimation_benchmark":
        return SimulationStandard.estimation_benchmark()
    if profile == "trajectory_tracking":
        return SimulationStandard.trajectory_tracking()
    raise ValueError(f"Unknown standard profile: {profile}")


def default_figure8_path(
    standard: SimulationStandard,
    *,
    alt: float = CRUISE_ALT,
    rx: float = 8.0,
    ry: float = 6.0,
) -> np.ndarray:
    """Generate a common figure-8 path with standard timing defaults."""
    return figure_8_path(
        duration=standard.duration,
        dt=0.1,
        alt=alt,
        alt_amp=0.0,
        rx=rx,
        ry=ry,
    )


def build_logger(
    sim_name: str,
    out_dir: Path,
    *,
    algorithm: str,
    standard: SimulationStandard,
    flight_coupled: bool,
    benchmark_mode: bool | None = None,
) -> SimLogger:
    """Create a logger with shared metadata wiring."""
    logger = SimLogger(sim_name, out_dir=out_dir)
    logger.log_metadata("algorithm", algorithm)
    logger.log_metadata("dt", standard.dt)
    logger.log_metadata("flight_coupled", flight_coupled)
    if benchmark_mode is not None:
        logger.log_metadata("benchmark_mode", benchmark_mode)
    return logger


def run_standard_composed_sim(
    *,
    sim_name: str,
    out_dir: Path,
    path: np.ndarray,
    standard: SimulationStandard,
    n_buildings: int = 0,
    world_size: float = 30.0,
    seed: int = 42,
    fallback_policy: str | None = None,
) -> RunResult:
    """One-call composed setup for quad mission runs."""
    env = create_environment(n_buildings=n_buildings, world_size=world_size, seed=seed)
    mission = create_mission(path=path, standard=standard, fallback_policy=fallback_policy)
    sim = create_sim(name=sim_name, out_dir=out_dir, mission=mission)
    platform = spawn_quad_platform()
    return run_sim(sim=sim, env=env, platform=platform)
