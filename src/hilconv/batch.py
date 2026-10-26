"""Batch capture and report replay without re-running physical solvers."""

import json
from pathlib import Path

from .experiments import ExperimentConfig, run
from .reporting import export_data, save_archive, write_json


def capture(configurations, directory, data_only=False, progress=None):
    configs = list(configurations)
    if not configs or any(not isinstance(c, ExperimentConfig) for c in configs):
        raise ValueError("nonempty list of validated experiment requests required")
    names = [c.case for c in configs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate cases would overwrite batch results")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for config in configs:
        result = run(config, progress)
        target = directory / config.case
        save_archive(result, target / "fields.npz")
        export_data(result, target)
        if not data_only:
            from .reporting import write_report

            write_report(result, target)
        entries.append(
            {
                "case": config.case,
                "archive": config.case + "/fields.npz",
                "configuration": result["configuration"],
                "provenance": result["provenance"],
                "summary": result["summary"],
            }
        )
        write_json(directory / "catalog.json", {"schema_version": 1, "experiments": entries})
    return directory / "catalog.json"


def load_catalog(directory):
    from .reporting import digest, load_archive

    directory = Path(directory).resolve()
    catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("unsupported batch catalog schema")
    entries = catalog.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("nonempty catalog required")
    results = []
    names = set()
    for entry in entries:
        config = ExperimentConfig.from_dict(entry["configuration"])
        if entry["case"] != config.case or config.case in names:
            raise ValueError("catalog case mismatch or duplicate")
        names.add(config.case)
        if entry["archive"] != config.case + "/fields.npz":
            raise ValueError("catalog requires canonical local archive paths")
        path = (directory / entry["archive"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("archive escapes catalog directory")
        result = load_archive(path)
        for key in ("configuration", "provenance", "summary"):
            if digest(entry[key]) != digest(result[key]):
                raise ValueError("catalog metadata disagrees with archive")
        if result["name"] != config.case:
            raise ValueError("archive case mismatch")
        results.append(result)
    return results


def render(directory, progress=None):
    from .reporting import verify_report, write_report

    results = load_catalog(directory)
    for result in results:
        if progress is not None:
            progress("Rendering " + result["name"])
        target = Path(directory) / result["name"]
        write_report(result, target)
        verify_report(target)
    return len(results)


def verify(directory):
    from .reporting import verify_report

    results = load_catalog(directory)
    for result in results:
        verify_report(Path(directory) / result["name"])
    return len(results)
