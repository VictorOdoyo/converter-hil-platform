# Reproduce and extend
Install uv and Python 3.11 or newer, then run from the checkout:

```powershell
uv sync --locked --extra dev
uv run hilconv list
uv run hilconv run --case buck-regulation --output outputs/buck
uv run hilconv run --case sinusoidal-inverter --output outputs/inverter
uv run hilconv batch --data-only --output outputs/catalog
uv run hilconv render outputs/catalog
uv run hilconv verify outputs/catalog
uv run hilconv replay outputs/catalog/buck-regulation/fields.npz --output outputs/replayed
uv run pytest -q
```

A request JSON contains only case, resolution_scale (integer 1..3) and seed (integer 0..2³²-1). Example: `{"case":"ideal-buck-ripple","resolution_scale":2,"seed":7}`. Invoke `run --config request.json --output outputs/request`; JSON cannot be mixed with scale/seed overrides.

Numerical scale refines independent plotted waveform/frequency grids. The exact-flow and sampled-control benches keep physical ADC/PWM clocks unchanged; there is no integration timestep to shrink. Some fixed-clock diagnostics therefore return the same values at every numerical scale.

The Python API exposes circuit.Circuit, bench.Config, bench.simulate and faults.Fault. Faults may change physical input/output source voltage, load resistance or resistive parasitics while energy-storage parameters stay fixed. Transport APIs accept encoded frames and deterministic virtual timestamps. Custom hardware integration would need a separate adapter and independent validation.

All fields.npz archives load with pickle disabled. Numeric arrays and JSON metadata are hashed independently; report manifests seal every delivered file. Render/replay preserve original solver provenance. Publish the entire gallery from one clean committed solve revision using `uv run python examples/build_gallery.py outputs/catalog`.
