import numpy as np
import pytest

from hilconv import core as m


def test_exact_endpoint():
    t = m.timeline(0.001, 0.0003)
    assert t[-1] == 0.001 and np.all(np.diff(t) > 0)
    assert len(t) == 5


def test_named_reproducible_streams():
    assert np.array_equal(m.stream(4, "adc").normal(size=8), m.stream(4, "adc").normal(size=8))
    assert not np.array_equal(m.stream(4, "adc").normal(size=8), m.stream(4, "link").normal(size=8))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_positive_real(value):
    with pytest.raises(ValueError):
        m.positive(value)
