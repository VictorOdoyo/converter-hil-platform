# Converter HIL Platform
[![Engineering checks](https://github.com/VictorOdoyo/converter-hil-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/VictorOdoyo/converter-hil-platform/actions/workflows/ci.yml)

A reproducible power-electronics laboratory for switched/averaged converter physics, sampled control and deterministic virtual HIL. Thirty studies connect exact switching intervals, ESR-aware energy balance, quantized measurements, delayed controllers and CRC-gated actuator commands.

```powershell
uv sync --locked --extra dev
uv run hilconv run --case buck-regulation --output outputs/buck
uv run hilconv run --case sinusoidal-inverter --output outputs/inverter
uv run hilconv run --case hil-wire-replay --output outputs/replay
uv run pytest -q
```

Power-stage benches cover buck, boost, bidirectional regeneration, full-bridge SPWM and diode discontinuous conduction. Closed-loop buck cases retain load/source faults, ADC bias, quantization, packet loss, corruption, watchdog blackouts and integrator windup. Independent design studies include boost nonminimum-phase response, discrete LQR and a voltage-only current observer.

[Computed results and full archives](docs/results.md) · [Architecture](docs/architecture.md) · [Circuit equations](docs/models.md) · [Control](docs/control.md) · [HIL protocol and timing](docs/timing.md) · [Experiments](docs/experiments.md) · [Reproduction](docs/reproduction.md) · [Validation](docs/validation.md) · [References](docs/references.md)

This is an offline synthetic platform. It needs no power hardware, FPGA or external service. Virtual deadlines do not demonstrate hard-real-time execution. Device switching loss and dead-time error are declared design approximations separate from the exact ideal-switch circuit model. MIT licensed.
