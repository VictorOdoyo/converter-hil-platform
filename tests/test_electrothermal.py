import numpy as np
import pytest

from hilconv import electrothermal as m


def test_device_parameters():
    assert m.Device().resistance_25_ohm == 0.02
    with pytest.raises(ValueError):
        m.Device(resistance_25_ohm=-1)


def test_thermal_parameters():
    assert m.Thermal().total_resistance == pytest.approx(3.7)
    with pytest.raises(ValueError):
        m.Thermal(case_capacity_j_k=0)


def test_resistance_temperature_law():
    assert m.resistance(25) == pytest.approx(0.02)
    assert m.resistance(125) == pytest.approx(0.028)


def test_rms_conduction():
    assert m.conduction_loss(10, 25) == pytest.approx(2)
    assert m.conduction_loss(20, 25) == pytest.approx(8)


def test_switching_energy_reference():
    assert m.switching_energy(10, 10, 48, 25) == pytest.approx(35e-6)
    assert m.switching_energy(0, 0, 48, 25) == 0


def test_two_node_thermal_energy():
    r = m.thermal_step([25, 25], 20, 25, 1)
    assert r["temperatures_c"][0] > r["temperatures_c"][1] > 25
    assert abs(r["energy_residual_j"]) < 1e-10


def test_thermal_equilibrium_fixed_point():
    equilibrium = m.steady_temperatures(20)
    assert np.allclose(m.thermal_step(equilibrium, 20, 25, 10)["temperatures_c"], equilibrium)


def test_steady_heat_budget():
    heat = m.allowable_heat(100, 25)
    assert m.steady_temperatures(heat)[0] == pytest.approx(100)
    assert m.allowable_heat(25, 50) == 0


def test_electrical_thermal_loss_partition():
    r = m.operating_point()
    assert r["device_heat_w"] > 0 and r["other_loss_w"] > 0
    assert r["input_w"] - r["output_w"] == pytest.approx(r["device_heat_w"] + r["other_loss_w"], abs=1e-8)
    assert abs(r["period_energy_residual_j"]) < 1e-10


def test_shutdown_hysteresis():
    latch = m.ThermalLatch()
    assert not latch.update(0, 95)
    assert not latch.update(1, 60)
    assert latch.update(2, 60)
    with pytest.raises(ValueError):
        latch.update(1, 80)
