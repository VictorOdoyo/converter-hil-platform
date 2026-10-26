import numpy as np
import pytest

from hilconv import circuit as m


def test_circuit_passivity():
    assert m.Circuit().input_v == 48
    with pytest.raises(ValueError):
        m.Circuit(capacitor_esr_ohm=-1)
    with pytest.raises(ValueError):
        m.state([1, 2, 3])


def test_kcl_esr():
    p = m.Circuit(capacitor_esr_ohm=0.2, load_source_v=10)
    r = m.ports([4, 20], 1, 1, p)
    assert 4 == pytest.approx(r["load_a"] + r["capacitor_a"])
    assert r["output_v"] - 20 == pytest.approx(0.2 * r["capacitor_a"])


def test_ideal_buck_slopes():
    p = m.Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    assert m.rhs([0, 24], 1, 1, p)[0] == pytest.approx(24 / p.inductance_h)
    assert m.rhs([0, 24], 0, 1, p)[0] == pytest.approx(-24 / p.inductance_h)


def test_instantaneous_energy_balance():
    p = m.Circuit(load_source_v=24)
    for sa, sb in [(1, 1), (1, 0), (0, 1), (-1, 1)]:
        x = np.array([-2.0, 18.0])
        derivative = m.rhs(x, sa, sb, p)
        w = m.power(x, sa, sb, p)
        de = p.inductance_h * x[0] * derivative[0] + p.capacitance_f * x[1] * derivative[1]
        assert de == pytest.approx(w["input_w"] - w["output_w"] - w["loss_w"], abs=1e-10)


def test_affine_generator():
    p = m.Circuit(load_source_v=12)
    a = m.generator(1, 0, p)
    x = np.array([2, 30, 1])
    assert np.allclose((a @ x)[:2], m.rhs(x[:2], 1, 0, p))
    assert (a @ x)[2] == 0


def test_power_quadratics():
    p = m.Circuit(load_source_v=24)
    z = np.array([-3, 18, 1.0])
    actual = np.einsum("i,kij,j->k", z, m.power_forms(1, 1, p), z)
    assert np.allclose(actual, list(m.power(z[:2], 1, 1, p).values()))


def test_exact_interval_energy():
    p = m.Circuit(load_source_v=12)
    x = np.array([4.0, 24.0])
    result = m.advance(x, 0.0001, 1, 1, p)
    change = m.energy(result["state"], p) - m.energy(x, p)
    assert change == pytest.approx(result["input_j"] - result["output_j"] - result["loss_j"], abs=2e-13)


def test_exact_vs_adaptive():
    p = m.Circuit(load_source_v=24)
    for sa, sb in [(1, 0), (0, 1), (-1, 1), (1, 1)]:
        a = m.advance([3, 20], 0.0003, sa, sb, p)["state"]
        b = m.reference_step([3, 20], 0.0003, sa, sb, p)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_random_signed_power_identities():
    rng = np.random.default_rng(291)
    for _ in range(100):
        p = m.Circuit(
            load_source_v=float(rng.uniform(0, 50)),
            input_v=float(rng.uniform(10, 80)),
            load_ohm=float(rng.uniform(0.1, 100)),
        )
        x = rng.uniform([-20, -40], [20, 80])
        sa = float(rng.choice([-1, 0, 1]))
        sb = float(rng.choice([0, 1]))
        derivative = m.rhs(x, sa, sb, p)
        power = m.power(x, sa, sb, p)
        rate = np.array([p.inductance_h * x[0], p.capacitance_f * x[1]]) @ derivative
        assert rate == pytest.approx(power["input_w"] - power["output_w"] - power["loss_w"], abs=1e-9)


def test_cached_flows_are_immutable():
    a = m.generator(1, 1)
    q = m.power_forms(1, 1)
    transition, _ = m.operators(tuple(a.ravel()), tuple(q.ravel()), 50e-6)
    with pytest.raises(ValueError):
        transition[0, 0] = 0
