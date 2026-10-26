from dataclasses import dataclass

import numpy as np

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


class CommandLatch:
    def __init__(self, maximum_age_ns=200000, watchdog_ns=300000, safe_duty=0.0):
        self.maximum_age = integer(maximum_age_ns, 0)
        self.watchdog = integer(watchdog_ns)
        self.safe = float(finite(safe_duty, real=True))
        if not 0 <= self.safe <= 1:
            raise ValueError("bounded safe duty required")
        self.sequence = None
        self.tick = None
        self.duty = self.safe
        self.accepted = 0
        self.rejected = 0
        self.time = None

    def receive(self, packet, now_ns):
        integer(now_ns, 0)
        try:
            f = decode(packet)
            if f.kind != 2 or len(f.values) != 1 or not 0 <= f.values[0] <= 1:
                raise ValueError("one bounded actuator duty required")
            if f.tick_ns > now_ns or now_ns - f.tick_ns > self.maximum_age:
                raise ValueError("future or stale command")
            if self.sequence is not None and not newer(f.sequence, self.sequence):
                raise ValueError("duplicate or reordered command")
        except ValueError:
            self.rejected += 1
            return False
        self.sequence = f.sequence
        self.tick = f.tick_ns
        self.duty = f.values[0]
        self.accepted += 1
        return True

    def output(self, now_ns):
        integer(now_ns, 0)
        if self.time is not None and now_ns < self.time:
            raise ValueError("monotone actuator clock required")
        self.time = now_ns
        stale = self.tick is None or now_ns - self.tick >= self.watchdog
        return {"duty": self.safe if stale else self.duty, "watchdog": stale}


class VirtualBus:
    def __init__(self, latency_ns=50000, jitter_ns=0, drop_every=0, corrupt_every=0, seed=0):
        from .core import stream

        self.latency = integer(latency_ns, 0)
        self.jitter = integer(jitter_ns, 0)
        self.drop = integer(drop_every, 0)
        self.corrupt = integer(corrupt_every, 0)
        self.rng = stream(seed, "transport")
        self.queue = []
        self.sent = 0
        self.dropped = 0
        self.time = None

    def send(self, packet, now_ns):
        import heapq

        integer(now_ns, 0)
        if not isinstance(packet, bytes):
            raise ValueError("encoded byte packet required")
        self.sent += 1
        if self.drop and self.sent % self.drop == 0:
            self.dropped += 1
            return None
        if self.corrupt and self.sent % self.corrupt == 0:
            data = bytearray(packet)
            data[-1] ^= 1
            packet = bytes(data)
        jitter = int(self.rng.integers(-self.jitter, self.jitter + 1)) if self.jitter else 0
        arrival = now_ns + max(0, self.latency + jitter)
        heapq.heappush(self.queue, (arrival, self.sent, packet))
        return arrival

    def poll(self, now_ns):
        import heapq

        integer(now_ns, 0)
        if self.time is not None and now_ns < self.time:
            raise ValueError("monotone virtual bus clock required")
        self.time = now_ns
        packets = []
        while self.queue and self.queue[0][0] <= now_ns:
            arrival, _, packet = heapq.heappop(self.queue)
            packets.append((arrival, packet))
        return packets


def deadlines(release_ns, execution_ns, period_ns):
    release = finite(release_ns, real=True, ndim=1)
    work = finite(execution_ns, real=True, ndim=1)
    period = integer(period_ns)
    if release.shape != work.shape or not len(release) or np.any(np.diff(release) <= 0) or np.any(work < 0):
        raise ValueError("ordered releases and aligned nonnegative execution budgets required")
    start = np.zeros(len(release))
    finish = np.zeros(len(release))
    previous = release[0]
    for k in range(len(release)):
        start[k] = max(release[k], previous)
        finish[k] = start[k] + work[k]
        previous = finish[k]
    return {
        "start_ns": start,
        "finish_ns": finish,
        "lateness_ns": np.maximum(0, finish - release - period),
        "missed": finish > release + period,
    }


def replay(time_ns, packet_bytes, received_ns, maximum_age_ns=200000, watchdog_ns=300000):
    times = np.asarray(time_ns)
    packets = np.asarray(packet_bytes)
    received = np.asarray(received_ns)
    if (
        times.dtype.kind not in "iu"
        or times.ndim != 1
        or len(times) < 1
        or np.any(np.diff(times) <= 0)
        or np.any(times < 0)
    ):
        raise ValueError("strictly ordered nonnegative integer replay ticks required")
    if (
        packets.dtype != np.uint8
        or packets.ndim != 2
        or packets.shape[1] != 32
        or received.dtype.kind not in "iu"
        or received.shape != (len(packets),)
        or np.any(np.diff(received) < 0)
    ):
        raise ValueError("ordered receive times and uint8 command frames required")
    if len(received) and (
        received[0] < times[0] or received[-1] > times[-1] or not np.all(np.isin(received, times))
    ):
        raise ValueError("receive timestamps must match recorded polling ticks")
    latch = CommandLatch(maximum_age_ns, watchdog_ns)
    index = 0
    duties = []
    stale = []
    for tick in times:
        now = int(tick)
        while index < len(packets) and received[index] <= now:
            latch.receive(packets[index].tobytes(), now)
            index += 1
        output = latch.output(now)
        duties.append(output["duty"])
        stale.append(output["watchdog"])
    return {
        "duty": np.array(duties),
        "watchdog": np.array(stale),
        "accepted": latch.accepted,
        "rejected": latch.rejected,
    }
