import numpy as np
import pytest

from hilconv import reporting as m


def test_provenance_hashes_actual_requests_and_preserves_frozen_source():
    from hilconv.results import curve, make

    config = {"case": "sample", "resolution_scale": 1, "seed": 42}
    frozen = m.provenance(config)
    result = make(
        "sample",
        {"input_v": 100},
        {"error": 0.01},
        {"x": [1.0, 2], "y": [0.1, 0.01]},
        [curve("Error", "x", [("y", "measured")], "h (m)", "Relative error")],
    )
    frozen["git_revision"] = "original-source"
    result = m.stamp(result, config, frozen)
    assert result["provenance"]["git_revision"] == "original-source"
    assert result["provenance"]["parameters_sha256"] == m.digest({"input_v": 100})
    assert m.digest(config) != m.digest({**config, "resolution_scale": 2})


def _example():
    from hilconv.results import curve, make

    return make(
        "sample",
        {"frequency_hz": 50},
        {"relative_error": 0.01},
        {"x": [1.0, 2.0, 3], "y": [0.1, 0.02, 0.01], "field": [1 + 2j, 3 + 4j]},
        [curve("Convergence", "x", [("y", "measured")], "h (m)", "Relative error")],
        {"samples": [{"x_m": 1.0, "error": 0.1}, {"x_m": 2.0, "error": 0.02}]},
    )


def test_archive_contains_numeric_fields_and_unicode_metadata_without_pickle(tmp_path):
    path = m.save_archive(_example(), tmp_path / "fields.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert np.iscomplexobj(archive["field"])
        assert archive["__metadata__"].dtype.kind == "U"


def test_field_archive_roundtrip_preserves_complex_values_and_source_revision(tmp_path):
    result = _example()
    config = {"case": "sample", "seed": 42}
    source = {**m.provenance(config), "git_revision": "saved-source"}
    result = m.stamp(result, config, source)
    restored = m.load_archive(m.save_archive(result, tmp_path / "fields.npz"))
    assert restored["provenance"] == result["provenance"]
    assert np.array_equal(restored["arrays"]["field"], result["arrays"]["field"])


def test_field_archive_rejects_changed_array_or_metadata(tmp_path):
    import json

    original = m.save_archive(_example(), tmp_path / "original.npz")
    with np.load(original, allow_pickle=False) as archive:
        copied = {key: archive[key].copy() for key in archive.files}
    copied["y"][0] = 0.9
    np.savez_compressed(tmp_path / "changed.npz", **copied)
    with pytest.raises(ValueError, match="array integrity"):
        m.load_archive(tmp_path / "changed.npz")
    meta = json.loads(str(copied["__metadata__"]))
    meta["summary"]["relative_error"] = 0.0
    copied["__metadata__"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp_path / "metadata.npz", **copied)
    with pytest.raises(ValueError, match="metadata integrity"):
        m.load_archive(tmp_path / "metadata.npz")


def test_exported_metadata_and_csv_have_canonical_lf_bytes_on_every_platform(tmp_path):
    import csv
    import json

    m.export_data(_example(), tmp_path)
    for path in tmp_path.iterdir():
        assert b"\r\n" not in path.read_bytes()
    with (tmp_path / "table_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[1]["error"]) == 0.02
    assert json.loads((tmp_path / "parameters.json").read_text())["frequency_hz"] == 50


def test_renderer_handles_curves_complex_components_and_masked_singular_regions(tmp_path):
    from types import SimpleNamespace

    from hilconv.results import image

    result = _example()
    grid = SimpleNamespace(x=np.arange(3) * 0.1, y=np.arange(3) * 0.1)
    result["arrays"]["map"] = np.arange(9).reshape(3, 3) * (1 + 1j)
    result["arrays"]["mask"] = np.eye(3, dtype=bool)
    result["plots"].append(image("Complex field magnitude", "map", grid, "|E| (V/m)", "abs", "mask"))
    path = m.plot_result(result, tmp_path / "figure.png")
    assert path.read_bytes().startswith(b"\x89PNG")


def test_html_report_escapes_metadata_and_records_all_physical_data(tmp_path):
    result = _example()
    result["summary"]["unsafe_label"] = "<script>bad()</script>"
    result["assumptions"] = ["<b>literal text</b>"]
    path = m.write_report(result, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "&lt;b&gt;" in text and "Physical parameters" in text
    assert (tmp_path / "fields.npz").is_file()
    assert (tmp_path / "table_samples.csv").is_file()
    assert b"\r\n" not in path.read_bytes()


def test_report_verification_detects_changed_tables_and_incomplete_manifests(tmp_path):
    import json

    m.write_report(_example(), tmp_path)
    assert m.verify_report(tmp_path)
    (tmp_path / "table_samples.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        m.verify_report(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"fields.npz": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        m.verify_report(tmp_path)
