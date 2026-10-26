# Circuit equations and energy
Let x=[i,vC]. The output is a Thevenin port containing source Vb in series with R. Capacitor ESR is rC, winding resistance is rL, inductance is L and ideal capacitance is C.

```text
Vout = (vC + rC*sb*i + rC*Vb/R)/(1+rC/R)
Iload = (Vout-Vb)/R
IC = sb*i-Iload
di/dt = (sa*Vin-sb*Vout-rL*i)/L
dvC/dt = IC/C
```

Buck modes use (sa,sb)=(gateA,1). Boost uses (1,1-gateA). Bidirectional uses (gateA,gateB). Full bridge uses (gateA-gateB,1). Common-carrier leg overlap is computed from actual edge intervals.

Stored energy is L*i²/2+C*vC²/2. Input power is sa*Vin*i; signed output-port power is Vout*Iload. Dissipation is rL*i²+rC*IC². Thus dE/dt=Pin-Pout-Ploss. Pout includes the external battery/source and the series resistor, so negative Pout can supply regenerative power. The resistor itself always dissipates nonnegative power.

Each held mode is affine: z' = A*z for z=[i,vC,1]. Powers are quadratic forms zᵀQz. The outer-product state evolves under A⊗I+I⊗A; an augmented exponential integrates all three powers over the same interval. Cached operators are immutable.

Averaging uses sums of mode fractions times vector fields and quadratic powers. Substituting a fractional sb before solving ESR algebra loses binary mode correlations; the boost ESR study demonstrates the resulting error.

Asynchronous buck/boost operation opens the diode at a bracketed current zero, advances isolated capacitor discharge and detects a subsequent forward-bias restart. Synchronous circuits permit negative current. The buck comparator inhibits positive on-pulses at its current threshold; it is not a symmetric negative-current limiter or galvanic isolation.

Switches are ideal in the solved circuit. Device switching power and average dead-time voltage error are separate illustrative estimates, not extra energy removed from these ideal-switch trajectories.
