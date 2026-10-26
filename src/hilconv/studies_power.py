from dataclasses import asdict

import numpy as np

from . import circuit, pwm
from .results import curve, make


def periodic(name, topology, da, db, model, scale):
    period = 50e-6
    x = pwm.periodic_state(period, topology, da, db, model)
    wave = pwm.waveform(x, period, topology, da, db, model, 401 * scale)
    r = pwm.cycle(x, period, topology, da, db, model)
    current = wave["state"][:, 0]
    voltage = wave["state"][:, 1]
    arrays = {
        "time_us": wave["time_s"] * 1e6,
        "current_a": current,
        "capacitor_v": voltage,
        "output_v": wave["output_v"],
        "source_coefficient": wave["source_coefficient"],
        "output_coefficient": wave["output_coefficient"],
    }
    summary = {
        "current_ripple_a": float(np.ptp(current)),
        "capacitor_ripple_v": float(np.ptp(voltage)),
        "input_w": r["input_j"] / period,
        "output_w": r["output_j"] / period,
        "loss_w": r["loss_j"] / period,
        "periodic_state_error": float(np.linalg.norm(r["state"] - x)),
    }
    return make(
        name,
        {"topology": topology, "duty_a": da, "duty_b": db, "period_s": period, "circuit": asdict(model)},
        summary,
        arrays,
        [
            curve(
                "Exact switched periodic current",
                "time_us",
                [("current_a", "Inductor")],
                "Carrier time (us)",
                "Current (A)",
            ),
            curve(
                "Capacitor versus ESR terminal voltage",
                "time_us",
                [("capacitor_v", "Capacitor"), ("output_v", "Terminal")],
                "Carrier time (us)",
                "Voltage (V)",
            ),
            curve(
                "Physical port switching coefficients",
                "time_us",
                [("source_coefficient", "Source"), ("output_coefficient", "Output")],
                "Carrier time (us)",
                "Signed coefficient",
            ),
        ],
    )


def buck(scale=1, seed=42):
    return periodic(
        "ideal-buck-ripple", "buck", 0.5, 0, circuit.Circuit(inductor_ohm=0, capacitor_esr_ohm=0), scale
    )


def boost(scale=1, seed=42):
    return periodic("boost-waveform", "boost", 0.4, 0, circuit.Circuit(load_ohm=12), scale)


def regeneration(scale=1, seed=42):
    return periodic(
        "bidirectional-regeneration",
        "bidirectional",
        0.4,
        1,
        circuit.Circuit(load_source_v=24, load_ohm=0.3),
        scale,
    )


def bridge(scale=1, seed=42):
    return periodic("full-bridge-waveform", "inverter", 0.75, 0.25, circuit.Circuit(), scale)


def dcm(scale=1, seed=42):
    from .hybrid import diode_cycle

    p = circuit.Circuit(load_ohm=100)
    period = 50e-6
    x = np.array([0.0, 24.0])
    states = [x.copy()]
    blocked = []
    energy = []
    for _ in range(120):
        r = diode_cycle(x, period, "buck", 0.15, p)
        x = r["state"]
        states.append(x.copy())
        blocked.append(r["blocked_s"] / period)
        energy.append(r["input_j"] - r["output_j"] - r["loss_j"])
    states = np.array(states)
    t = np.arange(len(states)) * period * 1000
    fraction = np.r_[blocked, blocked[-1]]
    residual = circuit.energy(x, p) - circuit.energy(states[0], p) - sum(energy)
    return make(
        "diode-discontinuous",
        {"circuit": asdict(p), "duty": 0.15},
        {
            "max_blocked_fraction": float(max(blocked)),
            "minimum_boundary_current_a": float(min(states[:, 0])),
            "energy_residual_j": float(residual),
        },
        {"time_ms": t, "current_a": states[:, 0], "capacitor_v": states[:, 1], "blocked_fraction": fraction},
        [
            curve(
                "Diode-constrained current",
                "time_ms",
                [("current_a", "Boundary current")],
                "Time (ms)",
                "Current (A)",
            ),
            curve(
                "Output capacitor discharge",
                "time_ms",
                [("capacitor_v", "Capacitor")],
                "Time (ms)",
                "Voltage (V)",
            ),
            curve(
                "Physical blocked time",
                "time_ms",
                [("blocked_fraction", "Fraction of carrier")],
                "Time (ms)",
                "Blocked fraction",
            ),
        ],
        assumptions=[
            "Synchronous models permit negative current; this asynchronous buck explicitly opens the diode at a zero crossing."
        ],
    )
