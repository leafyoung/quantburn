"""Python reference harness for quantburn parity tests.

Provides the SAME deterministic normal sampler as quantburn/src/rng.rs
(xorshift64* seeded PRNG + Box-Muller), so Rust and Python draw identical
standard normals. Conversion agents import this to generate expected values
for stochastic (Monte-Carlo / path-simulation) modules, then embed those
values in the Rust tests.

Usage inside an agent-generated reference script:

    import sys; sys.path.insert(0, "scripts")
    from refgen import normals  # normals(seed, n) -> list[float]
"""

import math


class Xorshift64Star:
    def __init__(self, seed: int):
        self.state = max(seed, 1)

    def next_u64(self) -> int:
        x = self.state & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 12)
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27)
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return (self.state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def next_f64(self) -> float:
        return (self.next_u64() >> 11) / float(1 << 53)

    def next_std_normal(self) -> float:
        u1 = max(self.next_f64(), 1e-300)
        u2 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def normals(seed: int, n: int) -> list:
    r = Xorshift64Star(seed)
    return [r.next_std_normal() for _ in range(n)]
