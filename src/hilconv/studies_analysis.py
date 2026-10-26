from dataclasses import asdict

import numpy as np

from .results import curve, make


def frequency(scale=1, seed=42):
    from . import control
    from .circuit import Circuit

    p = Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    f = np.geomspace(1, 20000, 501 * scale)
    arrays = {"frequency_hz": f}
    for name, duty in (("buck", 0.5), ("boost", 0.4)):
        response = control.frequency_response(control.linearize(name, duty, model=p), f)
        arrays[name + "_db"] = 20 * np.log10(abs(response))
        arrays[name + "_phase_deg"] = np.rad2deg(np.unwrap(np.angle(response)))
    return make(
        "duty-frequency-response",
        {"circuit": asdict(p), "boost_duty": 0.4},
        {"boost_rhp_zero_hz": p.load_ohm * 0.6**2 / (2 * np.pi * p.inductance_h)},
        arrays,
        [
            curve(
                "Duty-to-capacitor small-signal gain",
                "frequency_hz",
                [("buck_db", "Buck"), ("boost_db", "Boost")],
                "Frequency (Hz)",
                "Gain (dB V/duty)",
                xscale="log",
            ),
            curve(
                "Physical duty-response phase",
                "frequency_hz",
                [("buck_phase_deg", "Buck"), ("boost_phase_deg", "Boost")],
                "Frequency (Hz)",
                "Phase (deg)",
                xscale="log",
            ),
        ],
    )


def regulator(scale=1, seed=42):
    from . import control

    lin = control.linearize("buck", 0.5)
    period = 50e-6
    a, b = control.discrete_model(lin["a"], lin["b"], period)
    design = control.lqr(a, b, np.diag([0.01, 1]), np.array([[20.0]]))
    t = np.arange(401) * period
    x = np.array([0.0, 2.0])
    uncontrolled = x.copy()
    states = [x.copy()]
    opened = [x.copy()]
    commands = []
    for _ in t[:-1]:
        u = float((-design["gain"] @ x).item())
        commands.append(u)
        x = a @ x + b[:, 0] * u
        uncontrolled = a @ uncontrolled
        states.append(x.copy())
        opened.append(uncontrolled.copy())
    states = np.array(states)
    opened = np.array(opened)
    return make(
        "discrete-lqr",
        {"period_s": period, "gain": design["gain"].tolist(), "operating_state": lin["state"].tolist()},
        {
            "spectral_radius": float(max(abs(design["closed_loop_poles"]))),
            "final_voltage_error_v": float(states[-1, 1]),
        },
        {
            "time_ms": t * 1000,
            "closed_v": states[:, 1],
            "open_v": opened[:, 1],
            "duty_perturbation": np.r_[commands, commands[-1]],
        },
        [
            curve(
                "Small-signal voltage perturbation",
                "time_ms",
                [("closed_v", "LQR"), ("open_v", "Open loop")],
                "Time (ms)",
                "Voltage perturbation (V)",
            ),
            curve(
                "LQR duty perturbation",
                "time_ms",
                [("duty_perturbation", "State feedback")],
                "Time (ms)",
                "Duty deviation",
            ),
        ],
        assumptions=[
            "Linear perturbation benchmark, no saturation and no hardware deployment. Full state is intentionally available in this regulator design bench."
        ],
    )


def observer(scale=1, seed=42):
    from . import control
    from .core import stream

    lin = control.linearize("buck", 0.5)
    period = 50e-6
    a, b = control.discrete_model(lin["a"], lin["b"], period)
    h = np.array([[0.0, 1.0]])
    gain = control.observer_gain(a, h, [0.7, 0.8])
    rng = stream(seed, "observer")
    adc = control.ADC(14, -5, 5)
    truth = np.array([2.0, 1.0])
    estimate = np.zeros(2)
    t = np.arange(401) * period
    actual = []
    estimated = []
    measurements = []
    for _ in t:
        actual.append(truth.copy())
        estimated.append(estimate.copy())
        z = adc.convert(h @ truth + rng.normal(0, 0.002))["value"]
        measurements.append(float(z[0]))
        estimate = control.observer_step(estimate, z, [0], a, b, h, gain)
        truth = a @ truth
    actual = np.array(actual)
    estimated = np.array(estimated)
    tail = t > 0.01
    return make(
        "voltage-only-observer",
        {"adc_bits": 14, "voltage_noise_std_v": 0.002, "observer_gain": gain.tolist()},
        {"terminal_current_rmse_a": float(np.sqrt(np.mean((actual[tail, 0] - estimated[tail, 0]) ** 2)))},
        {
            "time_ms": t * 1000,
            "actual_a": actual[:, 0],
            "estimated_a": estimated[:, 0],
            "measured_v": np.array(measurements),
            "actual_v": actual[:, 1],
        },
        [
            curve(
                "Hidden inductor current reconstruction",
                "time_ms",
                [("actual_a", "Physical current"), ("estimated_a", "Voltage-only observer")],
                "Time (ms)",
                "Perturbation current (A)",
            ),
            curve(
                "Observer input",
                "time_ms",
                [("measured_v", "Quantized voltage"), ("actual_v", "Physical voltage")],
                "Time (ms)",
                "Voltage perturbation (V)",
            ),
        ],
    )


def deadtime(scale=1, seed=42):
    from .control import deadtime_error

    t = np.linspace(0, 0.04, 1001 * scale)
    current = 4 * np.sin(2 * np.pi * 50 * t)
    error = deadtime_error(current, 48, 1e-6, 20000)
    return make(
        "deadtime-voltage-error",
        {"bus_v": 48, "deadtime_s": 1e-6, "switching_hz": 20000},
        {"peak_error_v": float(max(abs(error)))},
        {"time_ms": t * 1000, "current_a": current, "error_v": error},
        [
            curve(
                "Continuous-current commutation sign",
                "time_ms",
                [("current_a", "Assumed current")],
                "Time (ms)",
                "Current (A)",
            ),
            curve(
                "Two-leg average dead-time voltage error",
                "time_ms",
                [("error_v", "Estimated error")],
                "Time (ms)",
                "Voltage error (V)",
            ),
        ],
        assumptions=[
            "Average polarity approximation with fixed current sign over commutation, not a switching transient or diode reverse-recovery solve."
        ],
    )


def deadline(scale=1, seed=42):
    from .transport import deadlines

    release = np.arange(100, dtype=np.int64) * 50000
    work = np.full(100, 25000, dtype=np.int64)
    work[30:36] = 90000
    r = deadlines(release, work, 50000)
    return make(
        "hil-deadline-overrun",
        {"period_ns": 50000, "ordinary_execution_ns": 25000, "overrun_execution_ns": 90000},
        {
            "missed_deadlines": int(sum(r["missed"])),
            "maximum_lateness_us": float(max(r["lateness_ns"]) / 1000),
        },
        {"release_ms": release / 1e6, "execution_us": work / 1000, "lateness_us": r["lateness_ns"] / 1000},
        [
            curve(
                "Declared execution budget",
                "release_ms",
                [("execution_us", "Execution")],
                "Release time (ms)",
                "Modeled work (us)",
            ),
            curve(
                "Backlog beyond deadline",
                "release_ms",
                [("lateness_us", "Lateness")],
                "Release time (ms)",
                "Modeled lateness (us)",
            ),
        ],
        assumptions=["All timing is virtual. This is not a host benchmark or hard-real-time certification."],
    )
