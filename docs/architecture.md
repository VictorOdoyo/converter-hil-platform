# Architecture
The power stage has two energy states: inductor current and ideal capacitor voltage. Each switch mode maps a source coefficient sa and an output coefficient sb onto one passive circuit. Buck, boost, a two-leg bidirectional stage and a full bridge share this formulation. Source and output power remain signed.

Exact affine matrix-exponential flows advance held modes. A lifted quadratic system integrates input energy, output-port energy and resistive loss. The event layer splits PWM edges and physical parameter faults, and separately resolves diode zero-current and positive-current comparator events.

The closed-loop bench is a synchronous buck with a voltage/current cascade. Quantized ADC observations pass through an integer-cycle delay. Computed duty commands travel through a binary CRC transport, virtual latency/jitter/loss and a timestamp watchdog before affecting the plant. Physical faults update only plant parameters; controller feedforward retains its nominal model.

The LQR, voltage-only observer, full-bridge SPWM and asynchronous-diode cases are independent design benches, with their assumptions recorded separately. No hardware drivers or hard-real-time runtime are required. See [equations](models.md), [timing](timing.md) and [validation](validation.md).
