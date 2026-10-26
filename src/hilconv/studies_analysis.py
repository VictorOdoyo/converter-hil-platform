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
