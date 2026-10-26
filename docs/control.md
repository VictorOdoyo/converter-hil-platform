# Sampled control and design benches
The main buck bench uses a voltage PI to request inductor current and a current PI to request duty. Saturation limits and back-calculation act at both stages. A soft-start reference, nominal load feedforward and nominal input-voltage feedforward are declared in the configuration. Actual input sag and load changes are not passed to this controller model.

Voltage/current channels have separate rails and an endpoint quantizer: code=round((clipped-lower)/LSB), LSB=(upper-lower)/(2^bits-1). Sensor gain/offset precede clipping. Seeded observation noise is independent from virtual transport jitter. A configurable measurement FIFO preserves sample causality.

PWM duty is quantized to integer timer counts, including exact off and full-on states. Controller integration uses elapsed seconds, and the physical carrier is retained on a shorter final interval rather than stretching PWM to fit the requested duration.

Independent small-signal benches linearize actual averaged vector fields, discretize with a zero-order hold, solve a discrete Riccati regulator and place observer poles. The observer receives voltage perturbation measurements only; the regulator bench intentionally assumes full-state feedback. Neither bench is silently substituted into the main PI experiment.

The boost duty-to-capacitor-voltage response includes the physical right-half-plane zero. The ideal reference is omega_z=R*(1-D)^2/L. The SPWM inverter bench switches a full bridge and fits low-order output harmonics after its startup transient. Its carrier-boundary harmonic fit excludes switching-band distortion.

The dead-time approximation is -Nlegs*Vin*deadtime*fs*sign(i) and assumes fixed current polarity over each commutation. The separate switching-loss estimate scales declared turn-on/off energies by current and voltage and adds Coss*V² per period. Both are illustrative design estimates.
