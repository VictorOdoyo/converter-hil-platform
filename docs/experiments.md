# Thirty study cases
Fifteen integrated buck cases cover nominal and averaged regulation, a load step, source sag, low-resistance short and near-open load, coarse ADC, unknown voltage offset, measurement delay, packet loss, CRC corruption, blackout, stale commands and unreachable references with/without anti-windup.

Nine power-stage studies cover ideal buck ripple, boost waveform, bidirectional regeneration, full-bridge switching, diode discontinuous conduction, switched/averaged convergence, ESR mode averaging, sinusoidal full-bridge output and illustrative switching loss.

Six analysis studies cover buck/boost frequency response, discrete LQR, voltage-only observation, average dead-time error, virtual deadline backlog and raw-wire command replay.

Exact case identifiers are listed by `uv run hilconv list`. All default runs and their inputs, arrays, summary CSVs, figures and verified reports appear in the [computed gallery](results.md). Adverse outcomes remain visible: a short cannot regulate, stale commands cannot actuate, and windup can retain a large tracking error.
