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
