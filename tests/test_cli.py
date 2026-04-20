"""
Tests for the meridianalgo CLI (meridianalgo.cli).
"""

import os
import sys

import pytest

os.environ["MERIDIANALGO_QUIET"] = "1"

from meridianalgo.cli import build_parser, cmd_info, cmd_version


class TestCLIParser:
    def test_parser_builds(self) -> None:
        parser = build_parser()
        assert parser is not None

    def test_version_subcommand_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, "_name_parser_map")
        )
        assert "version" in subparsers_action._name_parser_map

    def test_info_subcommand_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, "_name_parser_map")
        )
        assert "info" in subparsers_action._name_parser_map

    def test_demo_subcommand_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, "_name_parser_map")
        )
        assert "demo" in subparsers_action._name_parser_map

    def test_metrics_subcommand_registered(self) -> None:
        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if hasattr(a, "_name_parser_map")
        )
        assert "metrics" in subparsers_action._name_parser_map

    def test_no_command_has_no_func(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert not hasattr(args, "func")

    def test_metrics_default_period(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["metrics", "AAPL"])
        assert args.period == "2y"
        assert args.ticker == "AAPL"

    def test_metrics_custom_period(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["metrics", "MSFT", "--period", "1y"])
        assert args.period == "1y"


class TestCLICommands:
    def test_cmd_version_prints_version(self, capsys: pytest.CaptureFixture) -> None:
        from argparse import Namespace

        cmd_version(Namespace())
        captured = capsys.readouterr()
        assert "6.3.0" in captured.out

    def test_cmd_info_prints_modules(self, capsys: pytest.CaptureFixture) -> None:
        from argparse import Namespace

        cmd_info(Namespace())
        captured = capsys.readouterr()
        assert "core" in captured.out
        assert "MeridianAlgo" in captured.out


class TestCLIMain:
    def test_main_no_args_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from meridianalgo.cli import main

        monkeypatch.setattr(sys, "argv", ["meridianalgo"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    def test_main_version_subcommand(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from meridianalgo.cli import main

        monkeypatch.setattr(sys, "argv", ["meridianalgo", "version"])
        main()
        captured = capsys.readouterr()
        assert "6.3.0" in captured.out
