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


def test_bias():
    r = m.bias()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_delay():
    r = m.delay()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_loss():
    r = m.loss()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_corruption():
    r = m.corruption()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_blackout():
    r = m.blackout()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_stale():
    r = m.stale()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()


def test_antiwindup():
    r = m.antiwindup()
    assert r["summary"]["max_energy_residual_j"] < 1e-7
    assert r["arrays"]["time_s"][-1] == 0.04
    assert np.isfinite(r["arrays"]["state"]).all()
