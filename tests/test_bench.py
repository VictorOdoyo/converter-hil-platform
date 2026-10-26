import pytest

from hilconv import bench as m


def test_bench_config():
    assert m.Config().switching_hz == 20000
    with pytest.raises(ValueError):
        m.Config(adc_bits=25)
    with pytest.raises(ValueError):
        m.Config(blackout_s=(0.02, 0.01))


def test_integrated_buck_regulation():
    r = m.simulate(m.Config(duration_s=0.015))
    assert abs(r["output_v"][-1] - 24) < 1
    assert max(abs(r["balance_j"])) < 1e-9
    assert r["duty"][0] == 0
    assert r["accepted_packets"] > 100
