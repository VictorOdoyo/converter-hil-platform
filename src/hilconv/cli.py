"""Command line entry point for field studies and reproducible artifacts."""

import argparse
import json
from pathlib import Path

from . import __version__
from .experiments import CASES, ExperimentConfig, load_config, run


def parser():
    root = argparse.ArgumentParser(
        prog="hilconv", description="Reproducible converter and virtual HIL studies"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list available physical studies")
    single = commands.add_parser("run", help="solve one study and export a report")
    choice = single.add_mutually_exclusive_group(required=True)
    choice.add_argument("--case", choices=CASES)
    choice.add_argument("--config", type=Path)
    single.add_argument("--scale", type=int)
    single.add_argument("--seed", type=int)
    single.add_argument("--output", required=True, type=Path)
    batch = commands.add_parser("batch", help="solve multiple studies")
    batch.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    batch.add_argument("--scale", type=int, default=1)
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--data-only", action="store_true")
    batch.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render", help="render a captured catalog without solving")
    render.add_argument("directory", type=Path)
    replay = commands.add_parser("replay", help="render an archive while preserving source metadata")
    replay.add_argument("archive", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify report or catalog integrity")
    verify.add_argument("directory", type=Path)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "list":
            for case, (_, _, description) in CASES.items():
                print(f"{case:28} {description}")
        elif args.command == "run":
            from .reporting import write_report

            if args.config:
                if args.scale is not None or args.seed is not None:
                    raise ValueError("config file cannot be combined with scale or seed overrides")
                config = load_config(args.config)
            else:
                config = ExperimentConfig(
                    args.case, 1 if args.scale is None else args.scale, 42 if args.seed is None else args.seed
                )
            result = run(config, print)
            report = write_report(result, args.output)
            print(json.dumps(result["summary"], indent=2))
            print("Report: " + str(report.resolve()))
        elif args.command == "batch":
            from .batch import capture

            configs = [ExperimentConfig(case, args.scale, args.seed) for case in args.cases]
            print(capture(configs, args.output, args.data_only, print))
        elif args.command == "render":
            from .batch import render

            print(f"Rendered {render(args.directory, print)} studies")
        elif args.command == "replay":
            from .reporting import load_archive, write_report

            print(write_report(load_archive(args.archive), args.output))
        elif args.command == "verify":
            if (args.directory / "catalog.json").is_file():
                from .batch import verify

                print(f"Verified {verify(args.directory)} studies")
            else:
                from .reporting import verify_report

                verify_report(args.directory)
                print("Verified report")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        root.error(str(error))
    return 0
