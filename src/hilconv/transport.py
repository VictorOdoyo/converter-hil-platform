from dataclasses import dataclass

from .core import finite, integer


@dataclass(frozen=True)
class Frame:
    kind: int
    sequence: int
    tick_ns: int
    values: tuple

    def __post_init__(self):
        integer(self.sequence, 0)
        integer(self.tick_ns, 0)
        if self.sequence >= 2**32 or self.tick_ns >= 2**64 or self.kind not in (1, 2):
            raise ValueError("bounded wire sequence tick and frame kind required")
        finite(self.values, real=True, ndim=1)
        if not 1 <= len(self.values) <= 16:
            raise ValueError("one to sixteen declared SI values required")
