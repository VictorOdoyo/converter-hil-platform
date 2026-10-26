# Evidence and limits
Held-mode exact flows agree with an independent adaptive ODE solve. Randomized signed-state tests verify instantaneous stored-energy derivatives against input/output/loss power. Lifted quadratic integrals close finite-interval energy balances, including regenerative output sources, physical faults, diode blocking and comparator turn-off.

Other independent checks include ideal CCM buck/boost ratios, analytical inductor ripple, exact switched fixed points, switched/averaged frequency convergence and the ideal boost right-half-plane zero. The ESR counterexample distinguishes mode averaging from invalid fractional switch substitution.

Control tests cover ADC endpoints/rails, finite-word PWM, delayed observations, PI back-calculation, bumpless initialization, stable discrete LQR and voltage-only observer convergence. Integrated cases keep sensor bias as actual tracking error and reveal recovery limits rather than feeding truth into the controller.

Transport tests cover CRC corruption, length/version validation, immutable payloads, sequence rollover, unsigned replay clocks, stale/future/reordered commands, virtual-past injection and watchdog behavior. Raw-wire replay requires bit-for-bit equivalent applied duties and identical accepted/rejected counts.

Current extrema in closed-loop summaries use physical event samples. The positive-current comparator does not bound synchronous negative current. Energy residuals describe numerical consistency of this model, not measured converter efficiency. Dead-time and switching-loss estimates are not coupled semiconductor transients. Inverter low-order THD excludes switching-band distortion.

CI runs Python 3.11/3.13 on Linux/Windows, verifies the published gallery, checks formatting/lint and runs an installed wheel outside the checkout. These are synthetic offline studies, with no device datasheet calibration, magnetics saturation, parasitic switching ringing, electromagnetic interference certification, hardware-in-the-loop target, FPGA or hard-real-time guarantee.
