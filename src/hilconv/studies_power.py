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
    pulses = []
    for index in range(120):
        r = diode_cycle(x, period, "buck", 0.15, p)
        if index >= 117:
            pulses.extend(((index - 117) * period + row[0], row[1], row[2]) for row in r["trace"][1:])
        x = r["state"]
        states.append(x.copy())
        blocked.append(r["blocked_s"] / period)
        energy.append(r["input_j"] - r["output_j"] - r["loss_j"])
    states = np.array(states)
    t = np.arange(len(states)) * period * 1000
    fraction = np.r_[blocked, blocked[-1]]
    pulse = np.array(pulses)
    residual = circuit.energy(x, p) - circuit.energy(states[0], p) - sum(energy)
    return make(
        "diode-discontinuous",
        {"circuit": asdict(p), "duty": 0.15},
        {
            "max_blocked_fraction": float(max(blocked)),
            "minimum_boundary_current_a": float(min(states[:, 0])),
            "energy_residual_j": float(residual),
        },
        {
            "time_ms": t,
            "current_a": states[:, 0],
            "capacitor_v": states[:, 1],
            "blocked_fraction": fraction,
            "pulse_time_us": pulse[:, 0] * 1e6,
            "pulse_current_a": pulse[:, 1],
        },
        [
            curve(
                "Last three actual current pulses",
                "pulse_time_us",
                [("pulse_current_a", "Inductor current")],
                "Time (us)",
                "Current (A)",
            ),
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


def averaging(scale=1, seed=42):
    p = circuit.Circuit()
    frequencies = np.array([5000.0, 10000.0, 20000.0, 40000.0])
    duration = 0.002
    errors = []
    initial = np.array([4.0, 24.0])
    for f in frequencies:
        x = initial.copy()
        for _ in range(round(duration * f)):
            x = pwm.cycle(x, 1 / f, "buck", 0.5, model=p)["state"]
        reference = pwm.average_step(initial, duration, "buck", 0.5, model=p)["state"]
        errors.append(np.linalg.norm(x - reference))
    return make(
        "switched-average-convergence",
        {"duration_s": duration, "circuit": asdict(p)},
        {"coarse_error": errors[0], "fine_error": errors[-1]},
        {"frequency_hz": frequencies, "state_error": np.array(errors)},
        [
            curve(
                "Averaging at carrier boundaries",
                "frequency_hz",
                [("state_error", "Switched minus averaged state norm")],
                "Switching frequency (Hz)",
                "Mixed-unit state norm",
                xscale="log",
                yscale="log",
            )
        ],
        assumptions=[
            "Boundary current includes ripple phase; mixed-unit norm is only a convergence diagnostic."
        ],
    )


def esr(scale=1, seed=42):
    from scipy.linalg import expm

    values = np.linspace(0, 1, 101 * scale)
    correct = []
    naive = []
    difference = []
    x = np.array([10.0, 70.0])
    duration = 0.001
    for resistance in values:
        p = circuit.Circuit(capacitor_esr_ohm=float(resistance))
        a, _ = pwm.averaged("boost", 0.4, model=p)
        true = (expm(a * duration) @ np.r_[x, 1.0])[:2]
        wrong = (expm(circuit.generator(1, 0.6, p) * duration) @ np.r_[x, 1.0])[:2]
        correct.append(true[0])
        naive.append(wrong[0])
        difference.append(np.linalg.norm(true - wrong))
    return make(
        "esr-mode-averaging",
        {"duty": 0.4, "duration_s": duration},
        {"max_state_difference": float(max(difference))},
        {"esr_ohm": values, "correct_a": np.array(correct), "naive_a": np.array(naive)},
        [
            curve(
                "Average vector fields after switch algebra",
                "esr_ohm",
                [("correct_a", "Mode-fraction average"), ("naive_a", "Naive fractional output coefficient")],
                "Capacitor ESR (ohm)",
                "Inductor current (A)",
            )
        ],
    )


def inverter(scale=1, seed=42):
    from .metrics import harmonics

    p = circuit.Circuit()
    period = 50e-6
    t = np.arange(1201) * period
    x = np.zeros(2)
    states = [x.copy()]
    net = 0.0
    for time in t[:-1]:
        duty = (1 + 0.5 * np.sin(2 * np.pi * 50 * time)) / 2
        r = pwm.cycle(x, period, "inverter", duty, 1 - duty, p)
        x = r["state"]
        states.append(x.copy())
        net += r["input_j"] - r["output_j"] - r["loss_j"]
    states = np.array(states)
    reference = 24 * np.sin(2 * np.pi * 50 * t)
    valid = t >= 0.02
    fit = harmonics(t[valid], states[valid, 1], 50, 20)
    return make(
        "sinusoidal-inverter",
        {"carrier_hz": 20000, "output_hz": 50, "modulation_index": 0.5, "circuit": asdict(p)},
        {
            "fundamental_peak_v": float(fit["amplitude"][0]),
            "low_order_thd": fit["thd"],
            "energy_residual_j": circuit.energy(x, p) - net,
        },
        {
            "time_ms": t * 1000,
            "capacitor_v": states[:, 1],
            "current_a": states[:, 0],
            "reference_v": reference,
            "harmonic": np.arange(1, 21),
            "amplitude_v": fit["amplitude"],
        },
        [
            curve(
                "Filtered switched full bridge",
                "time_ms",
                [("capacitor_v", "Capacitor voltage"), ("reference_v", "Ideal fundamental")],
                "Time (ms)",
                "Voltage (V)",
            ),
            curve(
                "Actual inductor current", "time_ms", [("current_a", "Inductor")], "Time (ms)", "Current (A)"
            ),
            curve(
                "Low-order sampled harmonics",
                "harmonic",
                [("amplitude_v", "Least-squares fit")],
                "Harmonic order",
                "Peak voltage (V)",
                style="o",
            ),
        ],
        assumptions=[
            "Open-loop SPWM with exact switched intervals. Harmonic fit uses carrier-boundary samples and excludes switching-band distortion."
        ],
    )


def loss_estimate(scale=1, seed=42):
    current = np.linspace(0, 20, 101 * scale)
    loss20 = pwm.switching_loss(current, 48, 20000)
    loss80 = pwm.switching_loss(current, 48, 80000)
    return make(
        "switching-loss-estimate",
        {
            "bus_v": 48,
            "reference_current_a": 10,
            "turnon_j": 20e-6,
            "turnoff_j": 15e-6,
            "output_capacitance_f": 200e-12,
        },
        {"loss_20khz_at_20a_w": float(loss20[-1])},
        {"current_a": current, "loss20_w": loss20, "loss80_w": loss80},
        [
            curve(
                "Illustrative per-device switching power",
                "current_a",
                [("loss20_w", "20 kHz"), ("loss80_w", "80 kHz")],
                "Commutated current (A)",
                "Estimated power (W)",
            )
        ],
        assumptions=[
            "Illustrative energy scaling, not a device datasheet or thermal model. This estimate is not subtracted from the ideal-switch circuit energy a second time."
        ],
    )
