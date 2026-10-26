import pytest

from hilconv import bench as m


def test_bench_config():
    assert m.Config().switching_hz == 20000
    with pytest.raises(ValueError):
        m.Config(adc_bits=25)
    with pytest.raises(ValueError):
        m.Config(blackout_s=(0.02, 0.01))
