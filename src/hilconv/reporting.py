import numpy as np


def canonical(value):
    import json

    from .results import native

    return json.dumps(native(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    import hashlib

    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def provenance(configuration):
    import platform
    import subprocess
    from importlib.metadata import version
    from pathlib import Path

    from . import __version__

    root = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    if (root / "src" / "hilconv" / "__init__.py").is_file() and (root / "pyproject.toml").is_file():
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "package_version": __version__,
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "matplotlib": version("matplotlib"),
        "configuration_sha256": digest(configuration),
        "quantity_convention": "SI units; positive inductor current from source to load; signed input and output power; capacitor voltage excludes ESR; integer nanosecond HIL ticks",
    }


def stamp(result, configuration, source=None):
    from .results import native, validate

    source = provenance(configuration) if source is None else dict(source)
    if source["configuration_sha256"] != digest(configuration):
        raise ValueError("source configuration hash mismatch")
    parameter_hash = digest(result["parameters"])
    if "parameters_sha256" in source and source["parameters_sha256"] != parameter_hash:
        raise ValueError("source parameter hash mismatch")
    source["parameters_sha256"] = parameter_hash
    return validate({**result, "configuration": native(configuration), "provenance": source})


def _array_hash(array):
    import hashlib

    array = np.asarray(array)
    header = canonical({"dtype": array.dtype.str, "shape": array.shape}).encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _check_provenance(result):
    source = result["provenance"]
    if source["configuration_sha256"] != digest(result["configuration"]):
        raise ValueError("archived configuration hash mismatch")
    if source["parameters_sha256"] != digest(result["parameters"]):
        raise ValueError("archived physical parameter hash mismatch")


def save_archive(result, path):
    import os
    import tempfile
    from pathlib import Path

    from .results import native, validate

    validate(result)
    if "provenance" not in result:
        result = stamp(result, {"case": result["name"], "parameters": result["parameters"]})
    _check_provenance(result)
    arrays = {key: np.asarray(value) for key, value in result["arrays"].items()}
    metadata = native({key: value for key, value in result.items() if key != "arrays"})
    metadata["schema_version"] = 1
    metadata["array_sha256"] = {key: _array_hash(value) for key, value in arrays.items()}
    metadata["metadata_sha256"] = digest(metadata)
    arrays["__metadata__"] = np.array(canonical(metadata))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    return path


def load_archive(path):
    import json

    from .results import validate

    with np.load(path, allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            raise ValueError("missing field archive metadata")
        metadata = json.loads(str(archive["__metadata__"]))
        expected = metadata.pop("metadata_sha256", None)
        if expected != digest(metadata):
            raise ValueError("field archive metadata integrity mismatch")
        if metadata.pop("schema_version", None) != 1:
            raise ValueError("unsupported field archive schema")
        hashes = metadata.pop("array_sha256")
        if set(archive.files) != {*hashes, "__metadata__"}:
            raise ValueError("archive array inventory mismatch")
        arrays = {key: archive[key].copy() for key in hashes}
    for key, array in arrays.items():
        if _array_hash(array) != hashes[key]:
            raise ValueError("field archive array integrity mismatch: " + key)
    result = {**metadata, "arrays": arrays}
    _check_provenance(result)
    return validate(result)


def write_json(path, value):
    import json
    from pathlib import Path

    from .results import native

    Path(path).write_text(
        json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_data(result, directory):
    import csv
    from pathlib import Path

    from .results import validate

    validate(result)
    if "provenance" not in result:
        result = stamp(result, {"case": result["name"], "parameters": result["parameters"]})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("summary", result["summary"]),
        ("parameters", result["parameters"]),
        ("configuration", result["configuration"]),
        ("provenance", result["provenance"]),
    ):
        write_json(directory / (name + ".json"), value)
    tables = {"summary": [{"metric": key, "value": value} for key, value in result["summary"].items()]}
    tables.update({"table_" + key: value for key, value in result["tables"].items() if value})
    for name, rows in tables.items():
        if not rows:
            continue
        with (directory / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return directory


def plot_result(result, path):
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    from .results import validate

    validate(result)
    panels = result["plots"]
    columns = 1 if len(panels) == 1 else 2
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(7 * columns, 4.6 * rows), layout="constrained", squeeze=False
    )
    arrays = result["arrays"]
    for ax, plot in zip(axes.flat, panels, strict=False):
        if plot["kind"] == "curve":
            x = arrays[plot["x"]]
            selected = np.ones(len(x), bool) if plot["valid"] is None else arrays[plot["valid"]]
            for series in plot["series"]:
                y = np.asarray(arrays[series["key"]], dtype=float)
                series_valid = np.ones(len(x), bool) if "valid" not in series else arrays[series["valid"]]
                usable = selected & series_valid
                shown = np.where(usable, y, np.nan)
                style = plot.get("style", "o-" if len(x) <= 12 else "-")
                if any(word in series["label"].lower() for word in ("exact", "analytic", "reference")):
                    style = "--"
                ax.plot(
                    x,
                    shown,
                    style,
                    label=series["label"] + (" (unavailable)" if not np.any(usable) else ""),
                    linewidth=1.6,
                    markersize=4,
                )
            ax.set(xscale=plot["xscale"], yscale=plot["yscale"])
            if plot.get("ylim") is not None:
                ax.set_ylim(plot["ylim"])
            if "aspect" in plot:
                ax.set_aspect(plot["aspect"])
            ax.grid(alpha=0.22)
            ax.legend(fontsize=8)
        else:
            field = arrays[plot["array"]]
            if plot["component"] == "abs":
                field = abs(field)
            elif plot["component"] == "imag":
                field = field.imag
            else:
                field = field.real
            if plot["mask"] is not None:
                field = np.ma.masked_where(arrays[plot["mask"]], field)
            signed = bool(plot.get("signed", False))
            colormap = plt.get_cmap("RdBu_r" if signed else "viridis").with_extremes(bad="#d1d5db")
            limit = float(np.max(abs(field))) if signed else None
            shown = ax.pcolormesh(
                plot["x"],
                plot["y"],
                field,
                shading="nearest",
                cmap=colormap,
                vmin=-limit if signed else None,
                vmax=limit,
            )
            if plot.get("categorical", False):
                x, y = np.asarray(plot["x"]), np.asarray(plot["y"])
                ax.set(
                    xlim=(x[0] - (x[1] - x[0]) / 2, x[-1] + (x[-1] - x[-2]) / 2),
                    ylim=(y[0] - (y[1] - y[0]) / 2, y[-1] + (y[-1] - y[-2]) / 2),
                    xticks=x,
                    yticks=y,
                    aspect=plot.get("aspect", "auto"),
                )
            else:
                ax.set(
                    xlim=(plot["x"][0], plot["x"][-1]),
                    ylim=(plot["y"][0], plot["y"][-1]),
                    aspect=plot.get("aspect", "auto"),
                )
            fig.colorbar(shown, ax=ax, label=plot["colorbar"], pad=0.02)
        ax.set(title=plot["title"], xlabel=plot["xlabel"], ylabel=plot["ylabel"])
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(result["name"] + " | switched converters and virtual HIL", fontsize=14)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(result, directory):
    import hashlib
    import html
    from pathlib import Path

    from .results import native

    if "provenance" not in result:
        result = stamp(result, {"case": result["name"], "parameters": result["parameters"]})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_archive(result, directory / "fields.npz")
    export_data(result, directory)
    plot_result(result, directory / "figure.png")

    def escape(value):
        return html.escape(str(native(value)))

    summary_rows = "".join(
        "<tr><th>" + escape(key) + "</th><td>" + escape(value) + "</td></tr>"
        for key, value in result["summary"].items()
    )
    assumptions = "".join("<li>" + escape(item) + "</li>" for item in result["assumptions"])
    tables = "".join(
        '<li><a href="table_' + key + '.csv">' + escape(key) + " CSV</a></li>"
        for key, rows in result["tables"].items()
        if rows
    )
    title = escape(result["name"])
    source = result["provenance"]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Radar Perception Stack</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:3rem auto;padding:0 1rem;color:#172033}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.55rem;border-bottom:1px solid #dce1ea;text-align:left}}
code{{overflow-wrap:anywhere}}a{{color:#164ac4}}</style>
<h1>{title}</h1><p>Reproducible switched converters and virtual HIL. Units and field components are printed on the
scientific axes. peak d-q quantities use exp(+i omega t); terminal currents point into the protected line.</p>
<p><a href="fields.npz">Sealed numeric archive</a> | <a href="summary.csv">Summary CSV</a> |
<a href="parameters.json">Physical parameters</a> | <a href="configuration.json">Run request</a> |
<a href="provenance.json">Source and environment</a></p>
<img src="figure.png" alt="Scientific curves and electromagnetic field maps for {title}">
<table>{summary_rows}</table><h2>Data tables</h2><ul>{tables}</ul>
<h2>Assumptions and interpretation</h2><ul>{assumptions}</ul>
<p>These are simulations, not hardware measurements. Network-model, measurement, timing and statistical
uncertainty require distinct convergence checks. Singular or unilluminated data use explicit masks.</p>
<p>Source revision: <code>{escape(source["git_revision"])}</code>.
Dirty checkout: <code>{escape(source["git_dirty"])}</code>.
Configuration SHA-256: <code>{escape(source["configuration_sha256"])}</code>.</p>
<p>Replay preserves original source and parameter provenance. SHA-256 manifests detect changed bytes;
they are unsigned integrity records, not certificates of physical validity.</p></html>"""
    (directory / "index.html").write_text(document + "\n", encoding="utf-8", newline="\n")
    files = [
        "fields.npz",
        "figure.png",
        "summary.csv",
        "summary.json",
        "parameters.json",
        "configuration.json",
        "provenance.json",
        "index.html",
    ]
    files.extend("table_" + key + ".csv" for key, rows in result["tables"].items() if rows)
    manifest = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sorted(files)}
    write_json(directory / "manifest.json", manifest)
    return directory / "index.html"


def verify_report(directory):
    import hashlib
    import json
    import re
    from pathlib import Path

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "fields.npz",
        "figure.png",
        "summary.csv",
        "summary.json",
        "parameters.json",
        "configuration.json",
        "provenance.json",
        "index.html",
    }
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError("complete report manifest required")
    for name, expected in manifest.items():
        if (
            Path(name).name != name
            or "/" in name
            or "\\" in name
            or not isinstance(expected, str)
            or not re.fullmatch(r"[a-f0-9]{64}", expected)
        ):
            raise ValueError("manifest requires safe local filenames and SHA-256 digests")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("report integrity mismatch: " + name)
    result = load_archive(directory / "fields.npz")
    for key in ("summary", "parameters", "configuration", "provenance"):
        exported = json.loads((directory / (key + ".json")).read_text(encoding="utf-8"))
        if digest(exported) != digest(result[key]):
            raise ValueError("exported metadata disagrees with archive: " + key)
    for key, rows in result["tables"].items():
        if rows and "table_" + key + ".csv" not in manifest:
            raise ValueError("manifest omits an exported data table")
    return True
