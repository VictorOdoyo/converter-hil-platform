# Quasi-steady electrothermal extension
The standalone extension couples temperature-dependent semiconductor resistance to exact switched buck periodic equilibria. It does not replace the original fast sampled-control bench.

An equivalent conducting path adds Ron(T) to the physical inductor-series resistance. Exact quadratic moment integration determines RMS current and conduction heat; mean output/current and period energy are audited. Winding/capacitor dissipation is separated from semiconductor conduction. Turn-on and turn-off currents scale one declared illustrative commutation pair per carrier.

Commutation energy is represented as additional boundary input energy that is immediately dissipated. It changes total input/loss and thermal heat, but not the ideal-switch state trajectory. This is an explicit loss abstraction; it is not a switching-transient waveform or double-counted conduction energy.

Each slow thermal sample recomputes a frozen-temperature periodic electrical state and holds its heat over an exact junction-case-ambient Cauer step. The two-node thermal integrator also integrates ambient heat, closing its stored-energy balance. Electrical transients between equilibria are omitted, so only slow quasi-steady coupling is claimed.

The sampled temperature latch has trip/reset hysteresis and minimum off dwell. Its biased-sensor study retains higher actual junction temperatures when measured temperature under-reports them. Off-state electrical equilibrium is an abstraction; no power-hardware safety guarantee follows.

For modeling context see [Imperix thermal simulation](https://imperix.com/doc/help/thermal-simulation-with-imperix-power-library) and [Infineon junction-to-case thermal resistance](https://community.infineon.com/t5/Knowledge-Base-Articles/What-does-Junction-to-Case-thermal-resistance-mean/ta-p/616153). Parameters here are illustrative, not fitted device ratings.
