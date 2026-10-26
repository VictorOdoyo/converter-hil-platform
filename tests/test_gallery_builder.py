import runpy
from pathlib import Path

import pytest

from hilconv.batch import capture
from hilconv.experiments import ExperimentConfig


def test_publication_rejects_partial_catalog_before_creating_destination(tmp_path):
    builder = Path(__file__).resolve().parents[1] / "examples/build_gallery.py"
    source = tmp_path / "capture"
    destination = tmp_path / "gallery"
    capture([ExperimentConfig("deadtime-voltage-error")], source, data_only=True)
    with pytest.raises(ValueError, match="complete"):
        runpy.run_path(str(builder))["build"](source, destination)
    assert not destination.exists()
