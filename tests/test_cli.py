# Erwin Lejeune - 2026-02-18
"""Tests for the ``uav-sim`` command-line interface."""

import numpy as np
import pytest

from uav_sim.cli import catalogue
from uav_sim.cli.console import PALETTE, heading, style, supports_colour, table
from uav_sim.cli.main import build_parser, main


class TestCatalogue:
    def test_discovers_simulations(self):
        entries = catalogue.discover()
        assert len(entries) > 30
        assert all(entry.directory.joinpath("run.py").exists() for entry in entries)

    def test_slugs_are_unique(self):
        slugs = [entry.slug for entry in catalogue.discover()]
        assert len(slugs) == len(set(slugs))

    def test_entry_exposes_module_and_command(self):
        entry = catalogue.resolve("pid_hover")[0]
        assert entry.module == "uav_sim.simulations.path_tracking.pid_hover.run"
        assert entry.command.startswith("python -m uav_sim.simulations")

    def test_summary_comes_from_the_docstring(self):
        entry = catalogue.resolve("pid_hover")[0]
        assert entry.summary
        assert not entry.summary.startswith('"""')

    def test_summary_does_not_import_the_module(self, monkeypatch):
        """Listing must survive a simulation that fails to import."""
        import importlib

        def explode(_name):
            raise ImportError("boom")

        monkeypatch.setattr(importlib, "import_module", explode)
        assert catalogue.resolve("pid_hover")[0].summary

    def test_resolve_accepts_slug_name_and_glob(self):
        assert catalogue.resolve("path_tracking/pid_hover")[0].name == "pid_hover"
        assert catalogue.resolve("pid_hover")[0].name == "pid_hover"
        assert len(catalogue.resolve("swarm/*")) > 1

    def test_unknown_selector_suggests_alternatives(self):
        with pytest.raises(KeyError) as excinfo:
            catalogue.resolve("pid_hoverr")
        assert "Did you mean" in str(excinfo.value)

    def test_categories_partition_the_catalogue(self):
        grouped = catalogue.categories()
        assert sum(len(v) for v in grouped.values()) == len(catalogue.discover())


class TestConsole:
    def test_style_is_a_noop_without_a_tty(self):
        assert style("hello", "sky") == "hello"

    def test_style_emits_codes_when_supported(self, monkeypatch):
        monkeypatch.setattr("uav_sim.cli.console.supports_colour", lambda stream=None: True)
        assert PALETTE["sky"] in style("hello", "sky")

    def test_no_color_env_disables_colour(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")

        class FakeTTY:
            @staticmethod
            def isatty():
                return True

        assert not supports_colour(FakeTTY())

    def test_table_aligns_columns(self):
        """The widest cell in a column fixes that column's width for all rows."""
        rendered = table([("a", "bbbb"), ("cccc", "d")], headers=("x", "y"))
        _, first, second = rendered.splitlines()
        # "cccc" is the widest first-column cell, so column two starts at 6.
        assert first.index("bbbb") == 6
        assert second.index("d") == 6

    def test_table_of_nothing_is_empty(self):
        assert table([]) == ""

    def test_heading_includes_the_text(self):
        assert "Estimation" in heading("Estimation")


class TestParser:
    @pytest.mark.parametrize(
        "argv",
        [
            ["list"],
            ["run", "pid_hover"],
            ["info", "pid_hover"],
            ["envs"],
            ["train", "hover"],
            ["play", "hover"],
            ["trim"],
            ["doctor"],
        ],
    )
    def test_every_subcommand_is_wired(self, argv):
        args = build_parser().parse_args(argv)
        assert callable(getattr(args, "func", None))

    def test_bare_invocation_prints_help(self, capsys):
        assert main([]) == 0
        assert "uav-sim" in capsys.readouterr().out

    def test_train_defaults(self):
        args = build_parser().parse_args(["train", "hover"])
        assert args.optimizer == "ars"
        assert args.hidden is None


class TestCommands:
    def test_list_runs(self, capsys):
        assert main(["list"]) == 0
        assert "simulations" in capsys.readouterr().out

    def test_list_filters_by_category(self, capsys):
        assert main(["list", "--category", "estimation"]) == 0
        output = capsys.readouterr().out
        assert "ekf" in output
        assert "reynolds_flocking" not in output

    def test_list_rejects_an_unknown_category(self, capsys):
        assert main(["list", "--category", "nope"]) == 1
        assert "No category" in capsys.readouterr().err

    def test_info_prints_module_and_command(self, capsys):
        assert main(["info", "pid_hover"]) == 0
        output = capsys.readouterr().out
        assert "uav_sim.simulations.path_tracking.pid_hover" in output

    def test_info_on_unknown_simulation_fails(self, capsys):
        assert main(["info", "nonexistent"]) == 1
        assert "No simulation matches" in capsys.readouterr().err

    def test_envs_lists_every_environment(self, capsys):
        from uav_sim.gym import list_envs

        assert main(["envs"]) == 0
        output = capsys.readouterr().out
        for spec in list_envs():
            assert spec.env_id in output

    def test_trim_prints_a_table(self, capsys):
        assert main(["trim", "aerosonde"]) == 0
        output = capsys.readouterr().out
        assert "throttle" in output
        assert "stall speed" in output

    def test_trim_rejects_an_unknown_airframe(self, capsys):
        assert main(["trim", "spitfire"]) == 1
        assert "Unknown airframe" in capsys.readouterr().err

    def test_doctor_passes(self, capsys):
        assert main(["doctor"]) == 0
        output = capsys.readouterr().out
        assert "Physics self-check" in output
        assert "FAIL" not in output

    def test_train_end_to_end(self, capsys, tmp_path):
        destination = tmp_path / "policy.npz"
        code = main(
            [
                "train",
                "hover",
                "--iterations",
                "2",
                "--directions",
                "2",
                "--eval-episodes",
                "2",
                "--output",
                str(destination),
            ]
        )
        assert code == 0
        assert destination.exists()
        assert "Held-out evaluation" in capsys.readouterr().out

    def test_train_on_unknown_env_fails(self, capsys):
        assert main(["train", "nope", "--iterations", "1"]) == 1

    def test_play_with_the_untrained_policy(self, capsys):
        assert main(["play", "hover", "--episodes", "2"]) == 0
        output = capsys.readouterr().out
        assert "mean return" in output

    def test_play_a_saved_policy(self, capsys, tmp_path):
        from uav_sim.gym.policy import MLPPolicy

        policy = MLPPolicy(18, 4, hidden_sizes=(), seed=0)
        policy.parameters = np.random.default_rng(0).normal(0, 0.05, policy.parameter_count)
        path = policy.save(tmp_path / "p.npz")
        assert main(["play", "hover", "--policy", str(path), "--episodes", "1"]) == 0
