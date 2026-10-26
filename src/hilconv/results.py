import numpy as np


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def curve(
    title, x, series, xlabel, ylabel, xscale="linear", yscale="linear", valid=None, ylim=None, style=None
):
    return {
        "kind": "curve",
        "title": title,
        "x": x,
        "series": [
            {"key": item[0], "label": item[1], **({"valid": item[2]} if len(item) > 2 else {})}
            for item in series
        ],
        "xlabel": xlabel,
        "ylabel": ylabel,
        "xscale": xscale,
        "yscale": yscale,
        "valid": valid,
        "ylim": ylim,
        **({"style": style} if style is not None else {}),
    }


def image(title, array, grid, colorbar_label, component="real", mask=None):
    return {
        "kind": "image",
        "title": title,
        "array": array,
        "x": grid.x.tolist(),
        "y": grid.y.tolist(),
        "xlabel": "x (m)",
        "ylabel": "y (m)",
        "colorbar": colorbar_label,
        "component": component,
        "mask": mask,
    }


def make(name, parameters, summary, arrays, plots, tables=None, assumptions=()):
    result = {
        "name": name,
        "parameters": native(parameters),
        "summary": native(summary),
        "arrays": {key: np.asarray(value) for key, value in arrays.items()},
        "plots": native(plots),
        "tables": native(tables or {}),
        "assumptions": list(assumptions),
    }
    return validate(result)


def validate(result):
    import json
    import re

    if not isinstance(result["name"], str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", result["name"]):
        raise ValueError("safe hyphenated study name required")
    for group in ("arrays", "tables"):
        if not isinstance(result[group], dict) or any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) for key in result[group]
        ):
            raise ValueError("safe named data arrays and tables required")
    for array in result["arrays"].values():
        array = np.asarray(array)
        if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
            raise ValueError("study arrays must be finite numeric data")
    try:
        json.dumps(native({key: value for key, value in result.items() if key != "arrays"}), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("finite JSON-compatible metadata required; put complex fields in arrays") from error
    for rows in result["tables"].values():
        if not isinstance(rows, list):
            raise ValueError("tables must be lists of scalar records")
        if rows:
            columns = set(rows[0])
            for row in rows:
                if set(row) != columns or any(isinstance(value, (dict, list)) for value in row.values()):
                    raise ValueError("table records require consistent scalar columns")
    if not isinstance(result["plots"], list) or not result["plots"]:
        raise ValueError("at least one explicit scientific plot required")
    arrays = result["arrays"]
    for plot in result["plots"]:
        if plot["kind"] == "curve":
            x = np.asarray(arrays[plot["x"]])
            if x.ndim != 1 or x.size < 1 or np.iscomplexobj(x):
                raise ValueError("one-dimensional curve coordinates required")
            if plot.get("aspect", "auto") not in ("auto", "equal"):
                raise ValueError("curve aspect must be auto or equal")
            limits = plot.get("ylim")
            if limits is not None:
                limits = np.asarray(limits)
                if (
                    limits.shape != (2,)
                    or np.iscomplexobj(limits)
                    or not np.all(np.isfinite(limits))
                    or limits[0] >= limits[1]
                    or (plot["yscale"] == "log" and limits[0] <= 0)
                ):
                    raise ValueError("ordered finite real axis limits required")
            valid = np.ones(len(x), bool) if plot["valid"] is None else arrays[plot["valid"]]
            if valid.dtype != bool or valid.shape != x.shape:
                raise ValueError("nonempty explicit boolean spectral validity mask required")
            for axis in ("xscale", "yscale"):
                if plot[axis] not in ("linear", "log"):
                    raise ValueError("supported plot scales are linear and log")
            if plot["xscale"] == "log" and np.any(x[valid] <= 0):
                raise ValueError("positive logarithmic curve coordinates required")
            for series in plot["series"]:
                series_valid = np.ones(len(x), bool) if "valid" not in series else arrays[series["valid"]]
                if series_valid.dtype != bool or series_valid.shape != x.shape:
                    raise ValueError("aligned boolean series validity required")
                selected = valid & series_valid
                y = np.asarray(arrays[series["key"]])
                if y.shape != x.shape or np.iscomplexobj(y):
                    raise ValueError("curve ordinates must be real and aligned with x")
                if plot["yscale"] == "log" and np.any(y[selected] <= 0):
                    raise ValueError("logarithmic ordinates must be positive; record zeros explicitly")
        elif plot["kind"] == "image":
            field = arrays[plot["array"]]
            x, y = np.asarray(plot["x"]), np.asarray(plot["y"])
            if (
                field.shape != (len(y), len(x))
                or np.any(np.diff(x) <= 0)
                or np.any(np.diff(y) <= 0)
                or plot["component"] not in ("real", "imag", "abs")
            ):
                raise ValueError("image coordinates and selected field component must match")
            if plot["mask"] is not None:
                mask = arrays[plot["mask"]]
                if mask.dtype != bool or mask.shape != field.shape:
                    raise ValueError("image mask must be an aligned boolean array")
        else:
            raise ValueError("unknown scientific plot kind")
    return result
