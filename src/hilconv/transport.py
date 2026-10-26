from dataclasses import dataclass

from .core import finite, integer


@dataclass(frozen=True)
class Frame:
    kind: int
    sequence: int
    tick_ns: int
    values: tuple

    def __post_init__(self):
        integer(self.sequence, 0)
        integer(self.tick_ns, 0)
        if self.sequence >= 2**32 or self.tick_ns >= 2**64 or self.kind not in (1, 2):
            raise ValueError("bounded wire sequence tick and frame kind required")
        finite(self.values, real=True, ndim=1)
        if not 1 <= len(self.values) <= 16:
            raise ValueError("one to sixteen declared SI values required")


def encode(frame):
    import struct
    import zlib

    if not isinstance(frame, Frame):
        raise ValueError("validated HIL frame required")
    body = struct.pack(
        "<4sBBIQH", b"HIL1", 1, frame.kind, frame.sequence, frame.tick_ns, len(frame.values)
    ) + struct.pack("<" + "d" * len(frame.values), *frame.values)
    return body + struct.pack("<I", zlib.crc32(body))


def decode(packet):
    import struct
    import zlib

    if not isinstance(packet, bytes) or len(packet) < 32:
        raise ValueError("complete immutable HIL packet required")
    magic, version, kind, seq, tick, count = struct.unpack("<4sBBIQH", packet[:20])
    if magic != b"HIL1" or version != 1 or not 1 <= count <= 16 or len(packet) != 24 + 8 * count:
        raise ValueError("unsupported HIL header or length")
    if struct.unpack("<I", packet[-4:])[0] != zlib.crc32(packet[:-4]):
        raise ValueError("HIL CRC mismatch")
    return Frame(kind, seq, tick, struct.unpack("<" + "d" * count, packet[20:-4]))


def newer(sequence, previous):
    integer(sequence, 0)
    integer(previous, 0)
    if max(sequence, previous) >= 2**32:
        raise ValueError("32-bit sequence required")
    distance = (sequence - previous) % 2**32
    return 0 < distance < 2**31
