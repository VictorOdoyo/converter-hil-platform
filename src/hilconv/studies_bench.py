from dataclasses import asdict

import numpy as np

from . import bench, circuit
from .faults import Fault
from .results import curve, make


def solve(name, scale, seed, changes=None, events=(), model_changes=None):
    c = bench.Config(seed=seed, **(changes or {}))
    p = circuit.Circuit(**(model_changes or {}))
    faults = [Fault(*f) for f in events]
    r = bench.simulate(c, p, faults)
    arrays = {key: value for key, value in r.items() if isinstance(value, np.ndarray)}
    arrays["current_a"] = r["state"][:, 0]
    arrays["capacitor_v"] = r["state"][:, 1]
    arrays["time_ms"] = r["time_s"] * 1000
    terminal = r["time_s"] >= r["time_s"][-1] - 0.005
    summary = {
        "final_output_v": float(r["output_v"][-1]),
        "terminal_tracking_rms_v": float(
            np.sqrt(np.mean((r["output_v"][terminal] - r["reference_v"][terminal]) ** 2))
        ),
        "peak_inductor_a": float(max(abs(r["ripple_current_a"]))),
        "max_energy_residual_j": float(max(abs(r["balance_j"]))),
        "accepted_packets": r["accepted_packets"],
        "rejected_packets": r["rejected_packets"],
        "dropped_packets": r["dropped_packets"],
        "watchdog_cycles": r["watchdog_cycles"],
        "limited_cycles": int(sum(r["limited"][:-1])),
    }
    plots = [
        curve(
            "Physical output and ADC feedback",
            "time_ms",
            [("output_v", "Output"), ("measured_v", "Delayed ADC"), ("reference_v", "Reference")],
            "Time (ms)",
            "Voltage (V)",
        ),
        curve(
            "Inductor and cascade command",
            "time_ms",
            [("current_a", "Actual current"), ("current_reference_a", "Outer-loop current request")],
            "Time (ms)",
            "Current (A)",
        ),
        curve(
            "Requested and CRC-gated PWM",
            "time_ms",
            [("requested_duty", "Controller"), ("duty", "Applied")],
            "Time (ms)",
            "Duty fraction",
            ylim=(-0.05, 1.05),
        ),
        curve(
            "Exact interval energy residual",
            "time_ms",
            [("balance_j", "Stored - input + output + loss")],
            "Time (ms)",
            "Residual (J)",
            ylim=(-1e-7, 1e-7),
        ),
        curve(
            "Transport and comparator events",
            "time_ms",
            [("watchdog", "Watchdog"), ("limited", "Peak-current limiter")],
            "Time (ms)",
            "Active flag",
            ylim=(-0.1, 1.1),
        ),
        curve(
            "Cumulative signed energy",
            "time_ms",
            [
                ("input_j", "Input"),
                ("output_j", "Output port"),
                ("loss_j", "Resistive loss"),
                ("energy_j", "Stored"),
            ],
            "Time (ms)",
            "Energy (J)",
        ),
    ]
    return make(
        name,
        {"controller": c.to_dict(), "circuit": asdict(p), "faults": [asdict(f) for f in faults]},
        summary,
        arrays,
        plots,
        assumptions=[
            "Ideal switches; exact held-mode circuit flows and integrated quadratic power.",
            "Cascade PI consumes quantized delayed measurements; only plant parameters see physical faults.",
            "Virtual time transport and modeled deadlines do not demonstrate hardware or host real-time performance.",
            "Resolution scale preserves the physical sampling/PWM clock; exact interval physics has no integration step size.",
        ],
    )


def nominal(scale=1, seed=42):
    return solve("buck-regulation", scale, seed, {}, [])


def averaged(scale=1, seed=42):
    return solve("buck-averaged", scale, seed, {"averaged": True}, [])


def load_step(scale=1, seed=42):
    return solve("buck-load-step", scale, seed, {}, [[0.020013, "load_ohm", 3]])


def sag(scale=1, seed=42):
    return solve("buck-input-sag", scale, seed, {}, [[0.015013, "input_v", 36]])


def short(scale=1, seed=42):
    return solve("buck-output-short", scale, seed, {}, [[0.020013, "load_ohm", 0.2]])


def open_load(scale=1, seed=42):
    return solve("buck-load-open", scale, seed, {}, [[0.020013, "load_ohm", 1000000]])


def low_adc(scale=1, seed=42):
    return solve("low-resolution-adc", scale, seed, {"adc_bits": 7}, [])


def bias(scale=1, seed=42):
    return solve("biased-voltage-sensor", scale, seed, {"voltage_bias_v": 2}, [])


def delay(scale=1, seed=42):
    return solve("delayed-measurement", scale, seed, {"measurement_delay": 3}, [])


def loss(scale=1, seed=42):
    return solve("hil-packet-loss", scale, seed, {"drop_every": 3}, [])


def corruption(scale=1, seed=42):
    return solve("hil-corruption", scale, seed, {"corrupt_every": 3}, [])


def blackout(scale=1, seed=42):
    return solve("hil-blackout", scale, seed, {"blackout_s": [0.015, 0.018]}, [])


def stale(scale=1, seed=42):
    return solve("hil-stale-commands", scale, seed, {"latency_ns": 500000}, [])


def antiwindup(scale=1, seed=42):
    return solve("saturation-antiwindup", scale, seed, {"reference_v": 70, "reference_step_v": 24}, [])


def windup(scale=1, seed=42):
    return solve(
        "saturation-unprotected",
        scale,
        seed,
        {"reference_v": 70, "reference_step_v": 24, "antiwindup": False},
        [],
    )
