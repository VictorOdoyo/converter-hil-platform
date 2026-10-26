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


def cooling(scale=1, seed=42):
    from .core import stream
    from .electrothermal import cooling_fit

    t = np.linspace(0, 40, 101 * scale)
    truth = 25 + 40 * np.exp(-t / 12)
    measured = truth + stream(seed, "cooling").normal(0, 0.02, len(t))
    fit = cooling_fit(t, measured)
    reconstructed = 25 + fit["initial_excess_c"] * np.exp(-t / fit["time_constant_s"])
    return make(
        "cooling-time-identification",
        {"true_time_constant_s": 12, "sensor_std_c": 0.02},
        fit,
        {"time_s": t, "observed_c": measured, "fitted_c": reconstructed, "true_c": truth},
        [
            curve(
                "Observed one-lump cooling",
                "time_s",
                [("observed_c", "Measured"), ("fitted_c", "Fitted"), ("true_c", "Reference")],
                "Time (s)",
                "Temperature (C)",
            )
        ],
        assumptions=[
            "One-lump exponential identification benchmark; the two-node model is not mislabeled as single-exponential."
        ],
    )


def capacities(scale=1, seed=42):
    from .core import stream
    from .electrothermal import Thermal, capacity_fit, thermal_step

    p = Thermal()
    rng = stream(seed, "thermal-capacity")
    state = np.array([25.0, 25.0])
    delta = []
    energy = []
    states = [state.copy()]
    for _ in range(100 * scale):
        power = float(rng.uniform(0, 40))
        r = thermal_step(state, power, 25, 0.2, p)
        delta.append(r["temperatures_c"] - state)
        energy.append(r["input_j"] - r["ambient_j"])
        state = r["temperatures_c"]
        states.append(state.copy())
    fit = capacity_fit(delta, energy)
    truth = np.array([p.junction_capacity_j_k, p.case_capacity_j_k])
    return make(
        "thermal-capacity-identification",
        {"thermal_step_s": 0.2, "temperature_observations": "noise-free independent benchmark"},
        {
            "junction_capacity_j_k": float(fit["capacity_j_k"][0]),
            "case_capacity_j_k": float(fit["capacity_j_k"][1]),
            "max_fit_residual_j": float(max(abs(fit["residual_j"]))),
        },
        {
            "node_index": np.arange(2),
            "true_capacity_j_k": truth,
            "estimated_capacity_j_k": fit["capacity_j_k"],
            "delta_temperature_c": np.array(delta),
            "net_energy_j": np.array(energy),
        },
        [
            curve(
                "Independent node heat capacities",
                "node_index",
                [("true_capacity_j_k", "Reference"), ("estimated_capacity_j_k", "Fitted")],
                "Node index (junction=0, case=1)",
                "Heat capacity (J/K)",
                style="o",
            )
        ],
        assumptions=[
            "Both node temperatures and net heat flux are supplied. No capacity inference from one sensor or unknown losses is claimed."
        ],
    )


def arrhenius(scale=1, seed=42):
    from .electrothermal import arrhenius as factor

    t = np.linspace(25, 150, 101 * scale)
    arrays = {"temperature_c": t}
    for activation in (0.2, 0.3, 0.4):
        arrays["factor_" + str(int(activation * 10))] = factor(t, 25, activation)
    return make(
        "arrhenius-temperature-sensitivity",
        {"reference_c": 25, "activation_energies_ev": [0.2, 0.3, 0.4]},
        {"factor_at_150c_for_03ev": float(arrays["factor_3"][-1])},
        arrays,
        [
            curve(
                "Relative Arrhenius sensitivity",
                "temperature_c",
                [("factor_2", "0.2 eV"), ("factor_3", "0.3 eV"), ("factor_4", "0.4 eV")],
                "Temperature (C)",
                "Relative acceleration factor",
                yscale="log",
            )
        ],
        assumptions=[
            "Illustrative activation energies; no absolute lifetime, cycling fatigue or device qualification prediction."
        ],
    )


def feedback(scale=1, seed=42):
    from .electrothermal import Thermal, thermal_poles

    p = Thermal()
    slope = np.linspace(0, 1.5 / p.total_resistance, 121 * scale)
    poles = np.array([np.sort(thermal_poles(value, p).real) for value in slope])
    return make(
        "thermal-feedback-stability",
        {"total_thermal_resistance_k_w": p.total_resistance},
        {"critical_power_slope_w_k": 1 / p.total_resistance, "largest_test_pole_s_inv": float(poles[-1, -1])},
        {
            "power_slope_w_k": slope,
            "fast_pole": poles[:, 0],
            "slow_pole": poles[:, 1],
            "zero": np.zeros(len(slope)),
        },
        [
            curve(
                "Local temperature-power feedback poles",
                "power_slope_w_k",
                [("fast_pole", "Fast pole"), ("slow_pole", "Slow pole"), ("zero", "Stability boundary")],
                "dP/dT (W/K)",
                "Continuous pole real part (1/s)",
            )
        ],
        assumptions=[
            "Local linear power slope is supplied; it is not a global runaway certification or a physical failure simulation."
        ],
    )


def envelope(scale=1, seed=42):
    from .electrothermal import Thermal, allowable_heat

    ambient = np.linspace(0, 100, 101 * scale)
    arrays = {"ambient_c": ambient}
    for cooling in (1.0, 3.0, 6.0):
        model = Thermal(case_ambient_k_w=cooling)
        arrays["cooling_" + str(int(cooling))] = np.array(
            [allowable_heat(100, value, model) for value in ambient]
        )
    return make(
        "steady-thermal-budget",
        {"junction_limit_c": 100, "junction_case_k_w": 0.7},
        {"heat_budget_25c_default_w": allowable_heat(100, 25)},
        arrays,
        [
            curve(
                "Steady thermal design envelope",
                "ambient_c",
                [("cooling_1", "Rca=1 K/W"), ("cooling_3", "Rca=3 K/W"), ("cooling_6", "Rca=6 K/W")],
                "Ambient (C)",
                "Allowable constant device heat (W)",
            )
        ],
        assumptions=[
            "Declared synthetic temperature limit; no manufacturer safe-operating-area or transient current rating claim."
        ],
    )
