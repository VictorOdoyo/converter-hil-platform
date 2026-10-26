import numpy as np
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


def test_pipeline_transport_timing():
    r = m.simulate(m.Config(duration_s=0.001, latency_ns=100000))
    assert np.all(r["duty"][:2] == 0)
    log = r["packet_log"]
    assert np.all(log[:, 1] >= log[:, 0])
    assert r["accepted_packets"] > 0


def test_load_step_recovery_and_energy():
    from hilconv.faults import Fault

    r = m.simulate(m.Config(duration_s=0.03), faults=[Fault(0.015013, "load_ohm", 3)])
    assert abs(r["output_v"][-1] - 24) < 1
    assert max(abs(r["balance_j"])) < 1e-8
    assert 0.015013 in r["ripple_time_s"]


def test_blackout_watchdog():
    r = m.simulate(m.Config(duration_s=0.03, blackout_s=(0.01, 0.012)))
    selected = (r["time_s"] >= 0.0105) & (r["time_s"] < 0.012)
    assert np.all(r["watchdog"][selected] == 1)
    assert np.all(r["duty"][selected] == 0)
    assert abs(r["output_v"][-1] - 24) < 2


def test_corrupt_transport_retained():
    r = m.simulate(m.Config(duration_s=0.005, corrupt_every=2))
    assert r["rejected_packets"] > 20
    assert r["accepted_packets"] > 20
    assert np.isfinite(r["state"]).all()
    assert max(abs(r["balance_j"])) < 1e-9
