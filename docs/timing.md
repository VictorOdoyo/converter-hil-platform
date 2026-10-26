# Virtual HIL timing and wire protocol
At a carrier boundary the buck samples physical output/current, quantizes them, releases the delayed measurement, advances its PI controllers and sends a duty frame. The virtual bus releases arrived packets at the current polling tick. CRC, kind, length, duty bounds, timestamp age and unsigned sequence order gate the actuator latch. The watchdog then supplies the applied duty for the next held carrier interval.

The frame header is little-endian `<4sBBIQH`: magic HIL1, version 1, kind (1 measurement or 2 command), uint32 sequence, uint64 nanosecond tick and uint16 value count. One to sixteen finite float64 values follow, then CRC32 of the complete header and payload. A one-duty command occupies 32 bytes. Payloads are frozen on construction.

A newer sequence has modular distance in (0,2^31), including rollover. Future, duplicate, reordered, stale and corrupted frames are rejected. The main bench permits a maximum command age of 200 microseconds and watchdog age of 300 microseconds. A watchdog duty of zero turns the buck's high-side command off; this is a modeled actuator fallback, not a physical isolation switch.

Actual received bytes, virtual arrival, polling time and acceptance are archived. `hilconv.transport.replay` reconstructs the same latch decisions from those bytes and integer polling times, including CRC faults. The wire-replay study requires exact duty agreement.

Faults split physical integration at their configured timestamps, even between PWM edges. Coincident faults apply before the next interval; repeated same-time parameter updates retain input order. Floating-point carrier endpoint differences are coalesced.

The deadline study operates on declared execution budgets in virtual time and tracks backlog. No OS scheduling, FPGA deployment, external network, USB device, hardware target or wall-clock real-time capability is claimed.
