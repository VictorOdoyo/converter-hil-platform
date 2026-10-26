from dataclasses import dataclass

import numpy as np

from .core import finite, positive


@dataclass(frozen=True)
class Device:
    resistance_25_ohm: float = 0.02
    resistance_coefficient_k: float = 0.004
    turnon_reference_j: float = 20e-6
    turnoff_reference_j: float = 15e-6
    reference_current_a: float = 10
    reference_voltage_v: float = 48
    switching_temperature_coefficient_k: float = 0.002

    def __post_init__(self):
        for key in ("resistance_25_ohm", "reference_current_a", "reference_voltage_v"):
            positive(getattr(self, key))
        for key in (
            "resistance_coefficient_k",
            "turnon_reference_j",
            "turnoff_reference_j",
            "switching_temperature_coefficient_k",
        ):
            positive(getattr(self, key), allow_zero=True)


@dataclass(frozen=True)
class Thermal:
    junction_capacity_j_k: float = 0.5
    case_capacity_j_k: float = 10
    junction_case_k_w: float = 0.7
    case_ambient_k_w: float = 3

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            positive(getattr(self, key))

    @property
    def total_resistance(self):
        return self.junction_case_k_w + self.case_ambient_k_w


def resistance(junction_c, device=None):
    p = Device() if device is None else device
    t = finite(junction_c, real=True)
    if np.any(t <= -273.15):
        raise ValueError("physical junction temperature required")
    factor = 1 + p.resistance_coefficient_k * (t - 25)
    if np.any(factor <= 0):
        raise ValueError("temperature lies outside positive resistance model range")
    return p.resistance_25_ohm * factor


def conduction_loss(rms_current_a, junction_c, device=None):
    current = finite(rms_current_a, real=True)
    if np.any(current < 0):
        raise ValueError("nonnegative RMS current required")
    return current**2 * resistance(junction_c, device)


def switching_energy(turnon_current_a, turnoff_current_a, bus_v, junction_c, device=None):
    p = Device() if device is None else device
    voltage = positive(bus_v)
    temperature = float(finite(junction_c, real=True))
    if temperature <= -273.15:
        raise ValueError("physical commutation temperature required")
    factor = 1 + p.switching_temperature_coefficient_k * (temperature - 25)
    if factor < 0:
        raise ValueError("temperature outside switching-energy model")
    on = p.turnon_reference_j * abs(float(finite(turnon_current_a, real=True))) / p.reference_current_a
    off = p.turnoff_reference_j * abs(float(finite(turnoff_current_a, real=True))) / p.reference_current_a
    return (on + off) * voltage / p.reference_voltage_v * factor


def thermal_step(temperatures_c, power_w, ambient_c, duration_s, model=None):
    from scipy.linalg import expm

    p = Thermal() if model is None else model
    t = finite(temperatures_c, real=True)
    ambient = float(finite(ambient_c, real=True))
    power = positive(power_w, allow_zero=True)
    dt = positive(duration_s)
    if t.shape != (2,) or min(*t, ambient) <= -273.15:
        raise ValueError("junction case and ambient temperatures required")
    cj, cc, rj, ra = p.junction_capacity_j_k, p.case_capacity_j_k, p.junction_case_k_w, p.case_ambient_k_w
    a = np.array(
        [
            [-1 / (rj * cj), 1 / (rj * cj), power / cj, 0],
            [1 / (rj * cc), -(1 / rj + 1 / ra) / cc, ambient / (ra * cc), 0],
            [0, 0, 0, 0],
            [0, 1 / ra, -ambient / ra, 0],
        ]
    )
    result = expm(a * dt) @ np.r_[t, 1.0, 0.0]
    stored = float(np.array([cj, cc]) @ (result[:2] - t))
    return {
        "temperatures_c": result[:2],
        "input_j": power * dt,
        "ambient_j": float(result[3]),
        "stored_change_j": stored,
        "energy_residual_j": stored - power * dt + float(result[3]),
    }


def steady_temperatures(power_w, ambient_c=25, model=None):
    p = Thermal() if model is None else model
    power = positive(power_w, allow_zero=True)
    ambient = float(finite(ambient_c, real=True))
    if ambient <= -273.15:
        raise ValueError("physical ambient temperature required")
    return np.array([ambient + power * p.total_resistance, ambient + power * p.case_ambient_k_w])


def allowable_heat(maximum_junction_c, ambient_c=25, model=None):
    p = Thermal() if model is None else model
    maximum, ambient = finite([maximum_junction_c, ambient_c], real=True)
    if min(maximum, ambient) <= -273.15:
        raise ValueError("physical thermal limits required")
    return float(max(0, (maximum - ambient) / p.total_resistance))


def operating_point(junction_c=25, switching_hz=20000, duty=0.5, circuit_model=None, device=None):
    from dataclasses import replace

    from . import circuit, pwm

    p = circuit.Circuit(load_ohm=1) if circuit_model is None else circuit_model
    d = Device() if device is None else device
    frequency = positive(switching_hz)
    period = 1 / frequency
    if p.input_v <= 0 or p.load_source_v != 0:
        raise ValueError("positive-input resistively loaded buck operating point required")
    ron = float(resistance(junction_c, d))
    physical = replace(p, inductor_ohm=p.inductor_ohm + ron)
    initial = pwm.periodic_state(period, "buck", duty, model=physical)
    x = initial.copy()
    moments = np.zeros(3)
    energy = np.zeros(3)
    turnoff = float(initial[0])
    current = np.array([1.0, 0, 0])
    constant = np.array([0.0, 0, 1])
    den = 1 + physical.capacitor_esr_ohm / physical.load_ohm
    voltage = np.array([physical.capacitor_esr_ohm, 1.0, 0]) / den

    def symmetric(a, b):
        return (np.outer(a, b) + np.outer(b, a)) / 2

    forms = np.array(
        [symmetric(current, current), symmetric(voltage, constant), symmetric(current, constant)]
    )
    for lo, hi, sa, sb in pwm.segments("buck", duty):
        dt = (hi - lo) * period
        a = circuit.generator(sa, sb, physical)
        _, integrals = circuit.operators(tuple(a.ravel()), tuple(forms.ravel()), dt)
        z = np.r_[x, 1.0]
        moments += integrals @ np.outer(z, z).ravel()
        r = circuit.advance(x, dt, sa, sb, physical)
        x = r["state"]
        energy += np.array([r["input_j"], r["output_j"], r["loss_j"]])
        if sa:
            turnoff = float(x[0])
    rms = float(np.sqrt(max(0, moments[0] / period)))
    conduction = float(conduction_loss(rms, junction_c, d))
    switching = (
        float(switching_energy(initial[0], turnoff, p.input_v, junction_c, d) * frequency)
        if 0 < duty < 1
        else 0.0
    )
    pin, pout, loss = energy / period
    total_input = pin + switching
    return {
        "output_v": float(moments[1] / period),
        "rms_current_a": rms,
        "mean_current_a": float(moments[2] / period),
        "device_conduction_w": conduction,
        "switching_w": switching,
        "device_heat_w": conduction + switching,
        "other_loss_w": float(loss - conduction),
        "input_w": float(total_input),
        "output_w": float(pout),
        "efficiency": float(pout / total_input) if total_input > 1e-12 else 0.0,
        "period_energy_residual_j": float(energy[0] - energy[1] - energy[2]),
        "resistance_ohm": ron,
    }


class ThermalLatch:
    def __init__(self, trip_c=90, reset_c=75, minimum_off_s=2):
        self.trip, self.reset = finite([trip_c, reset_c], real=True)
        self.dwell = positive(minimum_off_s, allow_zero=True)
        if self.reset >= self.trip:
            raise ValueError("ordered trip and reset temperatures required")
        self.enabled = True
        self.changed = None
        self.time = None

    def update(self, time_s, measured_junction_c):
        time, temp = finite([time_s, measured_junction_c], real=True)
        if self.time is not None and time < self.time:
            raise ValueError("monotone thermal latch clock required")
        self.time = time
        if self.enabled and temp >= self.trip:
            self.enabled = False
            self.changed = time
        elif not self.enabled and temp <= self.reset and time - self.changed >= self.dwell:
            self.enabled = True
        return self.enabled


def simulate(
    duration_s=120,
    step_s=0.5,
    ambient_c=25,
    load_ohm=1,
    switching_hz=20000,
    shutdown=False,
    sensor_bias_c=0,
    thermal_model=None,
):
    from .circuit import Circuit
    from .core import timeline

    p = Thermal() if thermal_model is None else thermal_model
    time = timeline(duration_s, step_s)
    temperature = np.array([ambient_c, ambient_c], dtype=float)
    circuit = Circuit(load_ohm=positive(load_ohm))
    latch = ThermalLatch()
    states = [temperature.copy()]
    power = []
    output = []
    enabled = []
    residual = [0.0]
    total_residual = 0.0
    input_heat = [0.0]
    ambient_heat = [0.0]
    for start, dt in zip(time[:-1], np.diff(time), strict=True):
        on = latch.update(start, temperature[0] + sensor_bias_c) if shutdown else True
        op = operating_point(temperature[0], switching_hz, 0.5 if on else 0, circuit)
        advance = thermal_step(temperature, op["device_heat_w"], ambient_c, dt, p)
        temperature = advance["temperatures_c"]
        states.append(temperature.copy())
        total_residual += advance["energy_residual_j"]
        residual.append(total_residual)
        input_heat.append(input_heat[-1] + advance["input_j"])
        ambient_heat.append(ambient_heat[-1] + advance["ambient_j"])
        power.append(op["device_heat_w"])
        output.append(op["output_v"])
        enabled.append(on)
    return {
        "time_s": time,
        "temperatures_c": np.array(states),
        "device_heat_w": np.r_[power, power[-1]],
        "output_v": np.r_[output, output[-1]],
        "enabled": np.r_[enabled, enabled[-1]],
        "thermal_residual_j": np.array(residual),
        "input_heat_j": np.array(input_heat),
        "ambient_heat_j": np.array(ambient_heat),
    }


def cooling_fit(time_s, temperature_c, ambient_c=25):
    from .core import timebase

    t = timebase(time_s)
    temperature = finite(temperature_c, real=True, ndim=1)
    excess = temperature - float(finite(ambient_c, real=True))
    if t.shape != temperature.shape or np.any(excess <= 0):
        raise ValueError("aligned temperatures strictly above ambient required")
    slope, intercept = np.polyfit(t - t[0], np.log(excess), 1)
    if slope >= 0:
        raise ValueError("decaying thermal transient required")
    return {"time_constant_s": float(-1 / slope), "initial_excess_c": float(np.exp(intercept))}


def junction_from_case(case_c, power_w, model=None):
    p = Thermal() if model is None else model
    case = finite(case_c, real=True)
    power = positive(power_w, allow_zero=True)
    if np.any(case <= -273.15):
        raise ValueError("physical measured case temperature required")
    return case + power * p.junction_case_k_w


def capacity_fit(delta_temperatures, input_minus_ambient_j):
    delta = finite(delta_temperatures, real=True, ndim=2)
    energy = finite(input_minus_ambient_j, real=True, ndim=1)
    if delta.shape != (len(energy), 2) or len(energy) < 3 or np.linalg.cond(delta) > 1e8:
        raise ValueError("independently excited junction and case energy observations required")
    capacity = np.linalg.lstsq(delta, energy, rcond=None)[0]
    if np.any(capacity <= 0):
        raise ValueError("positive thermal capacities required")
    return {"capacity_j_k": capacity, "residual_j": energy - delta @ capacity}


def arrhenius(temperature_c, reference_c=25, activation_ev=0.3):
    temperature = finite(temperature_c, real=True) + 273.15
    reference = float(finite(reference_c, real=True)) + 273.15
    activation = positive(activation_ev)
    if np.any(temperature <= 0) or reference <= 0:
        raise ValueError("positive Kelvin temperatures required")
    exponent = activation / 8.617333262e-5 * (1 / reference - 1 / temperature)
    if np.any(abs(exponent) > 700):
        raise ValueError("temperature acceleration exceeds numerical range")
    return np.exp(exponent)
