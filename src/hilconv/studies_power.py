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
