def mode(topology, gate_a, gate_b=0):
    if gate_a not in (0, 1) or gate_b not in (0, 1):
        raise ValueError("binary gates required")
    if topology == "buck":
        return (gate_a, 1)
    if topology == "boost":
        return (1, 1 - gate_a)
    if topology == "bidirectional":
        return (gate_a, gate_b)
    if topology == "inverter":
        return (gate_a - gate_b, 1)
    raise ValueError("known converter topology required")
