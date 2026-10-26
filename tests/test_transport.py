import pytest

from hilconv import transport as m


def test_frame_validation():
    assert m.Frame(2, 0, 100, (0.5,)).values == (0.5,)
    with pytest.raises(ValueError):
        m.Frame(2, 2**32, 0, (0.5,))
    with pytest.raises(ValueError):
        m.Frame(1, 0, 0, (float("nan"),))


def test_wire_encoding():
    import struct
    import zlib

    data = m.encode(m.Frame(1, 7, 50000, (2.0, 24.0)))
    assert data[:6] == b"HIL1\x01\x01"
    assert struct.unpack("<I", data[-4:])[0] == zlib.crc32(data[:-4])
    assert len(data) == 40


def test_wire_roundtrip_corruption():
    frame = m.Frame(2, 2**32 - 1, 50000, (0.2, 0.7))
    data = m.encode(frame)
    assert m.decode(data) == frame
    bad = bytearray(data)
    bad[22] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        m.decode(bytes(bad))
    with pytest.raises(ValueError):
        m.decode(data + b"x")


def test_sequence_rollover():
    assert m.newer(0, 2**32 - 1)
    assert not m.newer(2**32 - 1, 0)
    assert not m.newer(3, 3)
    assert not m.newer(2**31, 0)
