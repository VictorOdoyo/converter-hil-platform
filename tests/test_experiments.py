import pytest

from hilconv.experiments import CASES, ExperimentConfig, load_config


@pytest.mark.parametrize(
    "value",
    [
        {"case": "unknown"},
        {"resolution_scale": True},
        {"resolution_scale": 1.5},
        {"resolution_scale": 0},
        {"resolution_scale": 4},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"extra": 1},
        [],
        None,
    ],
)
def test_request_rejects_ambiguous_or_unbounded_input(value):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(value)


def test_request_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"case":"deadtime-voltage-error","resolution_scale":2,"seed":7}', encoding="utf-8")
    config = load_config(path)
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert len(CASES) == 30


def test_dispatch_uses_physical_runner_and_preserves_request():
    from hilconv.experiments import run
    from hilconv.reporting import digest

    config = ExperimentConfig("deadtime-voltage-error", 1, 17)
    events = []
    result = run(config, events.append)
    assert result["name"] == config.case
    assert result["configuration"]["seed"] == 17
    assert result["provenance"]["configuration_sha256"] == digest(config.to_dict())
    assert len(events) == 1
    assert "parameters_sha256" in result["provenance"]


def test_all_catalog_runners_exist_without_running_heavy_simulations():
    from importlib import import_module

    for module, function, _ in CASES.values():
        assert callable(getattr(import_module("hilconv." + module), function))
