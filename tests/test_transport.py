import numpy as np
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


def test_command_latch_fail_closed():
    c = m.CommandLatch(100, 200)
    packet = m.encode(m.Frame(2, 0, 50, (0.7,)))
    assert not c.receive(packet, 40)
    assert c.receive(packet, 50)
    assert not c.receive(packet, 60)
    assert c.output(249)["duty"] == 0.7
    assert c.output(250)["duty"] == 0


def test_virtual_bus_causality_and_loss():
    b = m.VirtualBus(50, drop_every=2)
    assert b.send(b"data", 0) == 50
    assert b.send(b"drop", 1) is None
    assert not b.poll(49)
    assert b.poll(50) == [(50, b"data")]
    assert b.dropped == 1


def test_modeled_deadlines():
    r = m.deadlines([0, 100, 200], [150, 100, 10], 100)
    assert np.array_equal(r["start_ns"], [0, 150, 250])
    assert np.array_equal(r["missed"], [True, True, False])


def test_packet_replay_reproduces_corrupt_transport():
    from hilconv.bench import Config, simulate

    r = simulate(Config(duration_s=0.002, corrupt_every=3))
    ticks = np.rint(r["time_s"][:-1] * 1e9).astype(np.int64)
    replay = m.replay(ticks, r["received_packet_bytes"], r["packet_log"][:, 1])
    assert np.array_equal(replay["duty"], r["duty"][:-1])
    assert replay["accepted"] == r["accepted_packets"]
    assert replay["rejected"] == r["rejected_packets"]


def test_virtual_past_rejected():
    b = m.VirtualBus(50)
    b.poll(100)
    with pytest.raises(ValueError):
        m.validate_send_time(b, 99)
    assert m.validate_send_time(b, 100) == 100


def test_bus_send_calls_time_guard():
    b = m.VirtualBus(50)
    b.poll(100)
    with pytest.raises(ValueError):
        b.send(b"frame", 99)
