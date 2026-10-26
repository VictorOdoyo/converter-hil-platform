"""Validated reproducible run requests and lazy study dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

CASES = {
    "buck-regulation": [
        "studies_bench",
        "nominal",
        "Regulate a switched buck through quantized ADCs and one-cycle virtual HIL latency",
    ],
    "buck-averaged": [
        "studies_bench",
        "averaged",
        "Compare the same sampled cascade with an exactly averaged buck plant",
    ],
    "buck-load-step": [
        "studies_bench",
        "load_step",
        "Apply a load-resistance step at an asynchronous physical timestamp",
    ],
    "buck-input-sag": [
        "studies_bench",
        "sag",
        "Apply a physical input-voltage sag without revealing it to controller feedforward",
    ],
    "buck-output-short": [
        "studies_bench",
        "short",
        "Retain current-limited regulation failure after a finite low-resistance output fault",
    ],
    "buck-load-open": [
        "studies_bench",
        "open_load",
        "Propagate near-open output load dynamics and regenerative inductor current",
    ],
    "low-resolution-adc": [
        "studies_bench",
        "low_adc",
        "Exercise coarse ADC quantization in the physical voltage and current loops",
    ],
    "biased-voltage-sensor": [
        "studies_bench",
        "bias",
        "Retain voltage tracking error caused by an unknown ADC sensor offset",
    ],
    "delayed-measurement": [
        "studies_bench",
        "delay",
        "Expose control degradation from three-cycle measurement latency",
    ],
    "hil-packet-loss": [
        "studies_bench",
        "loss",
        "Hold and watchdog actuator commands during deterministic packet loss",
    ],
    "hil-corruption": [
        "studies_bench",
        "corruption",
        "Reject CRC-corrupt actuator packets in a closed physical feedback loop",
    ],
    "hil-blackout": [
        "studies_bench",
        "blackout",
        "Trip the actuator watchdog and recover after a timed communication blackout",
    ],
    "hil-stale-commands": [
        "studies_bench",
        "stale",
        "Fail closed when all transport commands exceed the declared age limit",
    ],
    "saturation-antiwindup": [
        "studies_bench",
        "antiwindup",
        "Recover from an unreachable voltage command using back-calculation",
    ],
    "saturation-unprotected": [
        "studies_bench",
        "windup",
        "Retain integrator windup during the same unreachable-reference challenge",
    ],
    "ideal-buck-ripple": ["studies_power", "buck", "Ideal buck ripple comparison"],
    "boost-waveform": [
        "studies_power",
        "boost",
        "Resolve boost inductor charging and output-current interruption within a carrier period",
    ],
    "bidirectional-regeneration": [
        "studies_power",
        "regeneration",
        "Demonstrate negative input power when an output-side battery regenerates through the converter",
    ],
    "full-bridge-waveform": [
        "studies_power",
        "bridge",
        "Resolve bipolar and zero bridge states with exact periodic current and ESR ripple",
    ],
    "diode-discontinuous": [
        "studies_power",
        "dcm",
        "Resolve light-load diode conduction and blocked intervals in a real switching trajectory",
    ],
    "switched-average-convergence": [
        "studies_power",
        "averaging",
        "Measure switched-cycle boundary error against physical frequency-averaging convergence",
    ],
    "esr-mode-averaging": [
        "studies_power",
        "esr",
        "Expose the energy and state error caused by averaging binary output coefficients before ESR algebra",
    ],
    "sinusoidal-inverter": [
        "studies_power",
        "inverter",
        "Generate filtered 50 Hz output from an actual switched unipolar full bridge",
    ],
    "switching-loss-estimate": [
        "studies_power",
        "loss_estimate",
        "Separate illustrative device switching-loss estimates from solved circuit conduction losses",
    ],
    "duty-frequency-response": [
        "studies_analysis",
        "frequency",
        "Compare buck and boost duty transfer functions including the boost right-half-plane zero",
    ],
    "discrete-lqr": [
        "studies_analysis",
        "regulator",
        "Compare discrete LQR perturbation dynamics with the unregulated physical linearization",
    ],
    "voltage-only-observer": [
        "studies_analysis",
        "observer",
        "Estimate hidden current from noisy quantized voltage perturbation measurements",
    ],
    "deadtime-voltage-error": [
        "studies_analysis",
        "deadtime",
        "Inspect sign-dependent full-bridge dead-time error under a declared continuous-current approximation",
    ],
    "hil-deadline-overrun": [
        "studies_analysis",
        "deadline",
        "Audit modeled CPU backlog and missed actuation deadlines without wall-clock claims",
    ],
    "hil-wire-replay": [
        "studies_analysis",
        "wire_replay",
        "Reproduce actuator decisions from raw CRC-corrupt and delayed command frames",
    ],
    "electrothermal-buck": [
        "studies_thermal",
        "nominal",
        "Couple switched conduction losses and semiconductor temperature during a slow thermal transient",
    ],
    "electrothermal-hot-ambient": [
        "studies_thermal",
        "hot",
        "Retain voltage and loss changes under a hotter thermal environment",
    ],
    "sampled-thermal-shutdown": [
        "studies_thermal",
        "shutdown",
        "Cycle a sampled thermal latch under an overloaded synthetic power stage",
    ],
    "biased-junction-sensor": [
        "studies_thermal",
        "bias",
        "Retain excess junction temperature when a biased sensor delays thermal shutdown",
    ],
    "thermal-switching-frequency": [
        "studies_thermal",
        "frequency",
        "Resolve switching-frequency loss and efficiency tradeoffs at a declared hot operating point",
    ],
    "electrothermal-loss-budget": [
        "studies_thermal",
        "budget",
        "Partition switched device conduction commutation winding and capacitor losses",
    ],
    "junction-case-step": [
        "studies_thermal",
        "step",
        "Compare exact two-node heating against steady limits and the limitations of case-only inference",
    ],
    "cooling-time-identification": [
        "studies_thermal",
        "cooling",
        "Identify a single-lump cooling time constant from noisy temperature observations",
    ],
    "thermal-capacity-identification": [
        "studies_thermal",
        "capacities",
        "Recover separate junction and case capacities from node measurements and independent energy flux",
    ],
    "arrhenius-temperature-sensitivity": [
        "studies_thermal",
        "arrhenius",
        "Compare relative temperature acceleration under explicitly illustrative activation energies",
    ],
    "thermal-feedback-stability": [
        "studies_thermal",
        "feedback",
        "Locate the two-node local thermal-feedback stability threshold",
    ],
    "steady-thermal-budget": [
        "studies_thermal",
        "envelope",
        "Compute declared steady junction heat budgets across ambient and cooling choices",
    ],
}


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = "buck-regulation"
    resolution_scale: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.case, str) or self.case not in CASES:
            raise ValueError("unknown experiment case")
        if type(self.resolution_scale) is not int or not 1 <= self.resolution_scale <= 3:
            raise ValueError("resolution_scale must be an integer from 1 to 3")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def to_dict(self):
        return {"case": self.case, "resolution_scale": self.resolution_scale, "seed": self.seed}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"case", "resolution_scale", "seed"}:
            raise ValueError("request must be an object containing only case, resolution_scale and seed")
        return cls(**value)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(config, progress=None):
    from importlib import import_module

    from .reporting import provenance, stamp
    from .results import validate

    if not isinstance(config, ExperimentConfig):
        raise ValueError("validated ExperimentConfig required")
    configuration = config.to_dict()
    source = provenance(configuration)
    module, function, _ = CASES[config.case]
    if progress is not None:
        progress("Solving " + config.case)
    result = getattr(import_module("hilconv." + module), function)(
        scale=config.resolution_scale, seed=config.seed
    )
    validate(result)
    if result["name"] != config.case:
        raise ValueError("study returned a different case name")
    return stamp(result, configuration, source)
