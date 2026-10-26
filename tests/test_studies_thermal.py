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


def test_cooling():
    assert abs(m.cooling()["summary"]["time_constant_s"] - 12) < 0.2


def test_capacities():
    s = m.capacities()["summary"]
    assert abs(s["junction_capacity_j_k"] - 0.5) < 1e-7 and abs(s["case_capacity_j_k"] - 10) < 1e-7


def test_arrhenius():
    assert m.arrhenius()["summary"]["factor_at_150c_for_03ev"] > 10


def test_feedback():
    assert m.feedback()["summary"]["largest_test_pole_s_inv"] > 0
