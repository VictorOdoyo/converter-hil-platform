import numpy as np

from hilconv import studies_bench as m


def test_nominal():
    r = m.nominal()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_averaged():
    r = m.averaged()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_load_step():
    r = m.load_step()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_sag():
    r = m.sag()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_short():
    r = m.short()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_open_load():
    r = m.open_load()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_low_adc():
    r = m.low_adc()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()
