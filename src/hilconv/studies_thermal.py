import numpy as np

from .results import curve, make


def transient(name, scale, seed, **changes):
    from .electrothermal import simulate

    r = simulate(**changes)
    temperature = r["temperatures_c"]
    arrays = {key: value for key, value in r.items() if isinstance(value, np.ndarray)}
    arrays["junction_c"] = temperature[:, 0]
    arrays["case_c"] = temperature[:, 1]
    return make(
        name,
        {"duration_s": 120, "thermal_step_s": 0.5, "carrier_hz": 20000, **changes},
        {
            "peak_junction_c": float(max(temperature[:, 0])),
            "final_junction_c": float(temperature[-1, 0]),
            "shutdown_samples": int(sum(~r["enabled"])),
            "max_thermal_energy_residual_j": float(max(abs(r["thermal_residual_j"]))),
        },
        arrays,
        [
            curve(
                "Junction and case thermal states",
                "time_s",
                [("junction_c", "Junction"), ("case_c", "Case")],
                "Time (s)",
                "Temperature (C)",
            ),
            curve(
                "Temperature-dependent device heat",
                "time_s",
                [("device_heat_w", "Conduction plus commutation")],
                "Time (s)",
                "Device heat (W)",
            ),
            curve(
                "Quasi-steady power-stage output",
                "time_s",
                [("output_v", "Output voltage")],
                "Time (s)",
                "Voltage (V)",
            ),
            curve(
                "Thermal energy closure",
                "time_s",
                [("thermal_residual_j", "Stored change - injected + ambient")],
                "Time (s)",
                "Residual (J)",
                ylim=(-1e-6, 1e-6),
            ),
        ],
        assumptions=[
            "Exact switched periodic electrical equilibrium at each frozen temperature, followed by held-power two-node thermal evolution.",
            "This is slow quasi-steady electrical/thermal coupling; electrical load transients between operating points are not resolved.",
            "Synthetic equivalent conducting path and one declared commutation pair per carrier; no device datasheet calibration.",
        ],
    )


def nominal(scale=1, seed=42):
    return transient("electrothermal-buck", scale, seed)


def hot(scale=1, seed=42):
    return transient("electrothermal-hot-ambient", scale, seed, ambient_c=55)


def shutdown(scale=1, seed=42):
    return transient("sampled-thermal-shutdown", scale, seed, load_ohm=0.5, shutdown=True)


def bias(scale=1, seed=42):
    return transient("biased-junction-sensor", scale, seed, load_ohm=0.5, shutdown=True, sensor_bias_c=-15)


def frequency(scale=1, seed=42):
    from .electrothermal import operating_point

    f = np.geomspace(5000, 100000, 61 * scale)
    heat = []
    efficiency = []
    switching = []
    for value in f:
        r = operating_point(100, float(value))
        heat.append(r["device_heat_w"])
        efficiency.append(r["efficiency"])
        switching.append(r["switching_w"])
    return make(
        "thermal-switching-frequency",
        {"junction_c": 100, "load_ohm": 1, "duty": 0.5},
        {"heat_at_5khz_w": heat[0], "heat_at_100khz_w": heat[-1]},
        {
            "frequency_khz": f / 1000,
            "heat_w": np.array(heat),
            "switching_w": np.array(switching),
            "efficiency_percent": 100 * np.array(efficiency),
        },
        [
            curve(
                "Periodic device loss at frozen temperature",
                "frequency_khz",
                [("heat_w", "Total device heat"), ("switching_w", "Commutation estimate")],
                "Carrier frequency (kHz)",
                "Power (W)",
                xscale="log",
            ),
            curve(
                "Electrical boundary efficiency",
                "frequency_khz",
                [("efficiency_percent", "Output / total input")],
                "Carrier frequency (kHz)",
                "Efficiency (%)",
                xscale="log",
            ),
        ],
        assumptions=[
            "Switching energy is an illustrative boundary loss; switching transients are not simulated."
        ],
    )


def budget(scale=1, seed=42):
    from .circuit import Circuit
    from .electrothermal import operating_point

    load = np.geomspace(0.5, 6, 61 * scale)
    conduction = []
    switching = []
    other = []
    balance = []
    for resistance in load:
        r = operating_point(75, circuit_model=Circuit(load_ohm=float(resistance)))
        conduction.append(r["device_conduction_w"])
        switching.append(r["switching_w"])
        other.append(r["other_loss_w"])
        balance.append(r["input_w"] - r["output_w"] - r["device_heat_w"] - r["other_loss_w"])
    return make(
        "electrothermal-loss-budget",
        {"junction_c": 75, "carrier_hz": 20000},
        {"maximum_power_residual_w": float(max(abs(np.array(balance))))},
        {
            "load_ohm": load,
            "conduction_w": np.array(conduction),
            "switching_w": np.array(switching),
            "other_w": np.array(other),
        },
        [
            curve(
                "Non-overlapping power loss partitions",
                "load_ohm",
                [
                    ("conduction_w", "Equivalent semiconductor conduction"),
                    ("switching_w", "Commutation estimate"),
                    ("other_w", "Winding and capacitor ESR"),
                ],
                "Load resistance (ohm)",
                "Power (W)",
                xscale="log",
            )
        ],
    )


def step(scale=1, seed=42):
    from .electrothermal import junction_from_case, steady_temperatures, thermal_step

    t = np.linspace(0, 240, 481 * scale)
    state = np.array([25.0, 25.0])
    states = [state.copy()]
    residual = []
    for dt in np.diff(t):
        r = thermal_step(state, 20, 25, dt)
        state = r["temperatures_c"]
        states.append(state.copy())
        residual.append(r["energy_residual_j"])
    states = np.array(states)
    equilibrium = steady_temperatures(20)
    inferred = junction_from_case(states[:, 1], 20)
    return make(
        "junction-case-step",
        {"power_w": 20, "ambient_c": 25},
        {
            "final_junction_c": float(state[0]),
            "steady_junction_c": float(equilibrium[0]),
            "max_energy_residual_j": float(max(abs(np.array(residual)))),
        },
        {
            "time_s": t,
            "junction_c": states[:, 0],
            "case_c": states[:, 1],
            "steady_junction_c": np.full(len(t), equilibrium[0]),
            "case_inferred_c": inferred,
        },
        [
            curve(
                "Two-node heating and steady prediction",
                "time_s",
                [
                    ("junction_c", "Actual junction"),
                    ("case_c", "Actual case"),
                    ("steady_junction_c", "Steady junction"),
                ],
                "Time (s)",
                "Temperature (C)",
            ),
            curve(
                "Steady-flow case inference during a transient",
                "time_s",
                [("junction_c", "Actual junction"), ("case_inferred_c", "Tc + P Rjc")],
                "Time (s)",
                "Temperature (C)",
            ),
        ],
        assumptions=[
            "Case-to-junction steady inference is deliberately shown to be biased during initial heating."
        ],
    )
