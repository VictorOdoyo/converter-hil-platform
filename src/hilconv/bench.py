from dataclasses import asdict, dataclass

import numpy as np

from . import circuit
from .core import finite, integer, positive


@dataclass(frozen=True)
class Config:
    duration_s: float = 0.04
    switching_hz: float = 20000
    reference_v: float = 24
    reference_step_v: float = 24
    reference_step_s: float = 0.025
    softstart_s: float = 0.003
    adc_bits: int = 12
    voltage_bias_v: float = 0
    measurement_delay: int = 0
    latency_ns: int = 50000
    jitter_ns: int = 0
    drop_every: int = 0
    corrupt_every: int = 0
    blackout_s: tuple = ()
    pwm_counts: int = 1000
    antiwindup: bool = True
    averaged: bool = False
    current_limit_a: float = 15
    seed: int = 42

    def __post_init__(self):
        for key in (
            "duration_s",
            "switching_hz",
            "reference_v",
            "reference_step_v",
            "softstart_s",
            "current_limit_a",
        ):
            positive(getattr(self, key))
        positive(self.reference_step_s, allow_zero=True)
        for key in ("measurement_delay", "latency_ns", "jitter_ns", "drop_every", "corrupt_every", "seed"):
            integer(getattr(self, key), 0)
        integer(self.adc_bits)
        integer(self.pwm_counts, 2)
        finite(self.voltage_bias_v, real=True)
        if (
            self.adc_bits > 24
            or self.seed >= 2**32
            or type(self.averaged) is not bool
            or type(self.antiwindup) is not bool
        ):
            raise ValueError("bounded resolution seed and boolean model switches required")
        if self.blackout_s:
            finite(self.blackout_s, real=True)
            if len(self.blackout_s) != 2 or not 0 <= self.blackout_s[0] < self.blackout_s[1]:
                raise ValueError("ordered blackout interval required")

    def to_dict(self):
        return asdict(self)


def simulate(config=None, model=None, faults=()):
    from . import control, plant, transport
    from .core import stream, timeline
    from .faults import Schedule

    c = Config() if config is None else config
    p = circuit.Circuit() if model is None else model
    positive(p.input_v, "nominal controller input voltage")
    schedule = Schedule(p, faults)
    period = 1 / c.switching_hz
    t = timeline(c.duration_s, period)
    n = len(t)
    voltage_adc = control.ADC(c.adc_bits, 0, 80, c.voltage_bias_v)
    current_adc = control.ADC(c.adc_bits, -30, 30)
    delay = control.Delay(c.measurement_delay, (0.0, 0.0))
    outer = control.PI(0.6, 100, -12, 12, 1000 if c.antiwindup else 0)
    inner = control.PI(0.008, 30, 0, 0.95, 2000 if c.antiwindup else 0)
    bus = transport.VirtualBus(c.latency_ns, c.jitter_ns, c.drop_every, c.corrupt_every, c.seed)
    latch = transport.CommandLatch(200000, 300000)
    rng = stream(c.seed, "adc")
    x = np.zeros(2)
    states = np.zeros((n, 2))
    states[0] = x
    keys = (
        "output_v",
        "raw_measured_v",
        "measured_v",
        "measured_a",
        "reference_v",
        "current_reference_a",
        "requested_duty",
        "duty",
        "watchdog",
        "limited",
        "clipped",
        "input_j",
        "output_j",
        "loss_j",
        "energy_j",
        "balance_j",
    )
    out = {key: np.zeros(n) for key in keys}
    trace = [(0.0, 0.0, 0.0)]
    clipped_count = 0
    watchdog_count = 0
    packet_log = []
    wire_packets = []
    for k, time in enumerate(t[:-1]):
        dt = t[k + 1] - time
        now = round(time * 1e9)
        physical = schedule.at(time)
        terminal = circuit.ports(x, 0, 1, physical)["output_v"]
        vr = voltage_adc.convert(terminal + rng.normal(0, 0.005))
        ir = current_adc.convert(x[0] + rng.normal(0, 0.005))
        mv, mi = delay.push((float(vr["value"]), float(ir["value"])))
        reference = (c.reference_v if time < c.reference_step_s else c.reference_step_v) * min(
            1, time / c.softstart_s
        )
        current_reference = outer.update(reference - mv, dt, reference / p.load_ohm)["command"]
        requested = inner.update(
            current_reference - mi, dt, mv / p.input_v + p.inductor_ohm * current_reference / p.input_v
        )["command"]
        quantized = float(control.timer_duty(requested, c.pwm_counts)["duty"])
        packet = transport.encode(transport.Frame(2, k % 2**32, now, (quantized,)))
        blackout = bool(c.blackout_s and c.blackout_s[0] <= time < c.blackout_s[1])
        if not blackout:
            bus.send(packet, now)
        for arrival, data in bus.poll(now):
            accepted = latch.receive(data, now)
            packet_log.append((arrival, now, int(accepted)))
            wire_packets.append(np.frombuffer(data, dtype=np.uint8).copy())
        command = latch.output(now)
        duty = command["duty"]
        r = plant.integrate_cycle(
            x,
            time,
            dt,
            period,
            "buck",
            duty,
            0,
            schedule,
            averaged=c.averaged,
            current_limit_a=None if c.averaged else c.current_limit_a,
        )
        x = r["state"]
        states[k + 1] = x
        trace.extend(map(tuple, r["trace"][1:]))
        values = {
            "output_v": terminal,
            "raw_measured_v": float(vr["value"]),
            "measured_v": mv,
            "measured_a": mi,
            "reference_v": reference,
            "current_reference_a": current_reference,
            "requested_duty": quantized,
            "duty": duty,
            "watchdog": int(command["watchdog"]),
            "limited": int(r["limited"]),
            "clipped": int(vr["clipped"] or ir["clipped"]),
        }
        for key, value in values.items():
            out[key][k] = value
        for key in ("input_j", "output_j", "loss_j"):
            out[key][k + 1] = out[key][k] + r[key]
        out["energy_j"][k + 1] = circuit.energy(x, p)
        out["balance_j"][k + 1] = (
            out["energy_j"][k + 1] - out["input_j"][k + 1] + out["output_j"][k + 1] + out["loss_j"][k + 1]
        )
        clipped_count += values["clipped"]
        watchdog_count += values["watchdog"]
    for key in keys:
        if not key.endswith("_j"):
            out[key][-1] = out[key][-2]
    out["output_v"][-1] = circuit.ports(x, 0, 1, schedule.at(t[-1]))["output_v"]
    ripple = np.array(trace)
    return {
        "time_s": t,
        "state": states,
        **out,
        "ripple_time_s": ripple[:, 0],
        "ripple_current_a": ripple[:, 1],
        "ripple_capacitor_v": ripple[:, 2],
        "packet_log": np.asarray(packet_log, dtype=np.int64).reshape(-1, 3),
        "received_packet_bytes": np.asarray(wire_packets, dtype=np.uint8).reshape(-1, 32),
        "accepted_packets": latch.accepted,
        "rejected_packets": latch.rejected,
        "dropped_packets": bus.dropped,
        "clipped_samples": clipped_count,
        "watchdog_cycles": watchdog_count,
        "configuration": c.to_dict(),
    }
