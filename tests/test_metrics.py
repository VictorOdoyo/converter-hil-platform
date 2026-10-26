import pytest

from hilconv import metrics as m


def test_time_weighting():
    r = m.statistics([0, 1, 3], [0, 2, 2])
    assert r["mean"] == pytest.approx(5 / 3)
    assert r["peak_to_peak"] == 2
