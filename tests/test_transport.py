import pytest

from hilconv import transport as m


def test_frame_validation():
    assert m.Frame(2, 0, 100, (0.5,)).values == (0.5,)
    with pytest.raises(ValueError):
        m.Frame(2, 2**32, 0, (0.5,))
    with pytest.raises(ValueError):
        m.Frame(1, 0, 0, (float("nan"),))
