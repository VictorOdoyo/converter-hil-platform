from hilconv import studies_thermal as m


def test_nominal():
    r = m.nominal()
    assert r["summary"]["max_thermal_energy_residual_j"] < 1e-6
    assert r["summary"]["peak_junction_c"] > 25


def test_hot():
    r = m.hot()
    assert r["summary"]["max_thermal_energy_residual_j"] < 1e-6
    assert r["summary"]["peak_junction_c"] > 25


def test_shutdown():
    r = m.shutdown()
    assert r["summary"]["max_thermal_energy_residual_j"] < 1e-6
    assert r["summary"]["peak_junction_c"] > 25


def test_bias():
    r = m.bias()
    assert r["summary"]["max_thermal_energy_residual_j"] < 1e-6
    assert r["summary"]["peak_junction_c"] > 25


def test_frequency():
    s = m.frequency()["summary"]
    assert s["heat_at_100khz_w"] > s["heat_at_5khz_w"]


def test_budget():
    assert m.budget()["summary"]["maximum_power_residual_w"] < 1e-7


def test_step():
    s = m.step()["summary"]
    assert abs(s["final_junction_c"] - s["steady_junction_c"]) < 0.5 and s["max_energy_residual_j"] < 1e-9
