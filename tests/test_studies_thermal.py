from hilconv import studies_thermal as m


def test_nominal():
    r = m.nominal()
    assert r["summary"]["max_thermal_energy_residual_j"] < 1e-6
    assert r["summary"]["peak_junction_c"] > 25


def test_hot():
    r = m.hot()
    assert r["summary"]["max_thermal_energy_residual_j"] < 1e-6
    assert r["summary"]["peak_junction_c"] > 25
