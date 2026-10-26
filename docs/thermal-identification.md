# Thermal validation and identification
Twelve additional studies cover nominal and hot ambient coupling, overload shutdown, biased temperature sensing, switching-frequency tradeoffs, non-overlapping power-loss budgets, exact two-node heating, cooling time identification, two-node capacity identification, Arrhenius sensitivity, local thermal feedback and steady cooling budgets.

The constant-power thermal step converges to Tj=Ta+P*(Rjc+Rca), Tc=Ta+P*Rca. Case-only Tj=Tc+P*Rjc inference is valid for steady flow and is deliberately shown to be biased during startup. Cooling regression is separately tested on a single-lump exponential, not misapplied to a two-pole transient.

Capacity identification uses independently supplied changes in both node temperatures and net input-minus-ambient energy. Both channels must be excited. The noise-free identification benchmark does not imply identifiability from one sensor and unknown losses.

The thermal feedback bench accepts a local dP/dT and checks its continuous poles; its critical slope is 1/(Rjc+Rca). This is a local linear diagnostic, not a global runaway certification. Arrhenius factors use declared illustrative activation energies and give relative sensitivity only, without absolute lifetime or fatigue predictions.

All studies remain offline and synthetic. See [electrothermal assumptions](electrothermal-coupling.md) and the [computed gallery](results.md).
