# quantburn

Rust/Burn port of [`../quantorch/torchquantlib`](../quantorch) (the
Python/PyTorch quant pricing library), converted per-file: one Rust module per
Python source file, with embedded `#[cfg(test)]` parity tests against Python
reference values. `cargo test --lib` runs **121 tests**, all green, all within
floating-point tolerance (1e-6 … 1e-14 depending on the module — see the few
relaxed cases noted in test comments).

This is a numerics port, not a framework port: where torchquantlib relies on
PyTorch autograd/optimizer training loops (calibration harnesses, seq2seq PDE
solver, GAN training), only the portable numeric core was converted and the
training machinery was left out. All prices are plain `f64` math or Burn
tensor ops — nothing here trains models.

## Coverage overview

**Options — closed form / trees**

- `black_scholes_merton` — BSM closed form, European and the Python-source
  American approximation.
- `american_option`, `bermudan_option`, `binomial_tree_model`,
  `option_utils` (shared CRR binomial-tree engine).
- `letsberational` — "Let's Be Rational" implied-volatility algorithm.

**Options — exotics & Monte Carlo**

- `exotics` — the exotic payoff suite ported from `exotics.py` (barrier,
  Asian, lookback, cliquet, rainbow, spread, snowball, etc.).
- `malliavin` — Malliavin-calculus Monte Carlo Greeks.
- `gans` — numeric forward-pass core (linear/activations/BCE) of the GAN
  module; no training loop.

**Stochastic models**

- `heston`, `sabr`, `dupire_local_volatility`, `stochastic_model`,
  `binomial_tree_model`.

**Interest-rate models**

- `vasicek`, `cir`, `hull_white`, `black_karasinski`, `lmm` (Libor Market
  Model).

**Bonds & fixed income**

- `bond_pricer` (zero-coupon, coupon, stochastic-rate, callable, convertible),
  `fixed_income_forward`, `future_pricer`.

**Credit risk**

- `structural_model` (Merton), `reduced_form_model`.

**Market risk**

- `var`, `expected_shortfall`, `scenario_analysis`, `stress_testing`,
  `greeks`, `valuation_adjustment` (xVA), `yield_curve_construction`.

**Infrastructure**

- `rng` — deterministic xorshift64* + Box–Muller normal sampler,
  bit-exact parity with the Python harness.

## Not converted (no portable numeric core)

- `calibration/*` — PyTorch optimizer/training harnesses only.
- `utils/seq2seq_pde_solver.py` — seq2seq network training harness.
- `models/stochastic_volatility/gans.py` — training loop only; its
  forward-pass numeric core is ported in `src/gans.rs`.

## Quick start

```rust
use quantburn::black_scholes_merton::black_scholes_merton;

// European call, S=100 K=105 T=1 r=5% q=2% sigma=20%
let price = black_scholes_merton("call", "european", 100.0, 105.0, 1.0, 0.2, 0.05, 0.02);
println!("{price:.6}"); // 6.986920
```

Scalars are `f64` function arguments (no tensors needed for closed-form
math); simulation-based modules take an explicit `seed: u64` and draw normals
from the deterministic sampler, so results are reproducible across runs and
identical to the Python reference generators.

Runnable counterparts of the Python `examples/` live in `examples/`
(`cargo run --example american_option`, `cargo run --example heston_option_pricing`, …).

## Python file → Rust module mapping

| torchquantlib (Python) | quantburn (Rust) |
|---|---|
| `core/asset_pricing/option/american_option.py` | `src/american_option.rs` |
| `core/asset_pricing/option/asian_option.py` | `src/asian_option.rs` |
| `core/asset_pricing/option/bermudan_option.py` | `src/bermudan_option.rs` |
| `core/asset_pricing/option/black_scholes_merton.py` | `src/black_scholes_merton.rs` |
| `core/asset_pricing/option/exotics.py` | `src/exotics.rs` |
| `core/asset_pricing/option/utils.py` | `src/option_utils.rs` |
| `core/asset_pricing/letsberational.py` | `src/letsberational.rs` |
| `core/asset_pricing/bond/bond_pricer.py` | `src/bond_pricer.rs` |
| `core/asset_pricing/bond/fixed_income_forward.py` | `src/fixed_income_forward.rs` |
| `core/asset_pricing/money/future_pricer.py` | `src/future_pricer.rs` |
| `models/binomial_tree.py` | `src/binomial_tree_model.rs` |
| `models/stochastic_model.py` | `src/stochastic_model.rs` |
| `models/stochastic_volatility/heston.py` | `src/heston.rs` |
| `models/stochastic_volatility/sabr.py` | `src/sabr.rs` |
| `models/stochastic_volatility/gans.py` (forward core only) | `src/gans.rs` |
| `models/local_volatility/dupire_local_volatility.py` | `src/dupire_local_volatility.rs` |
| `models/interest_rate/vasicek.py` | `src/vasicek.rs` |
| `models/interest_rate/cir.py` | `src/cir.rs` |
| `models/interest_rate/hull_white.py` | `src/hull_white.rs` |
| `models/interest_rate/black_karasinski.py` | `src/black_karasinski.rs` |
| `models/interest_rate/lmm.py` | `src/lmm.rs` |
| `greeks/greeks.py` | `src/greeks.rs` |
| `greeks/malliavin.py` | `src/malliavin.rs` |
| `credit_risk/structural_model.py` | `src/structural_model.rs` |
| `credit_risk/reduced_form_model.py` | `src/reduced_form_model.rs` |
| `market_risk/var.py` | `src/var.rs` |
| `market_risk/expected_shortfall.py` | `src/expected_shortfall.rs` |
| `market_risk/scenario_analysis.py` | `src/scenario_analysis.rs` |
| `market_risk/stress_testing.py` | `src/stress_testing.rs` |
| `valuation_adjustment/valuation_adjustment.py` | `src/valuation_adjustment.rs` |
| `utils/yield_curve_construction.py` | `src/yield_curve_construction.rs` |
| (no Python counterpart) | `src/rng.rs` — shared deterministic sampler |

## Parity-test methodology

- `scripts/refgen.py` is the Python twin of `src/rng.rs`
  (xorshift64* + Box–Muller); both produce bit-identical standard normals.
- `scripts/gen_<module>.py` are pure-float64 Python reference generators: they
  run the original torchquantlib numerics (torch RNG replaced by the shared
  sampler) and print expected values, which are embedded as literals in the
  Rust `#[cfg(test)]` modules.
- Because MC paths consume the same deterministic normals in both languages,
  parity is checked on the full simulation pipeline in float64 — not by
  matching torch's RNG.
- Tolerance policy: closed-form modules use tight bounds (typically 1e-9 to
  1e-14); Monte-Carlo and iteration-based modules use looser bounds (1e-6 to
  1e-8), with any relaxation documented in the test's comment.

## Benchmarks

`bench/` contains a 4-backend cross-validation and benchmark suite comparing
torch CPU, torch ROCm (HIP), Burn NdArray (CPU) and Burn Wgpu/Vulkan on ten
shared workloads (BSM grids, implied vol, binomial trees, MC European/Asian,
Heston/CIR simulation, FD Greeks, VaR/ES, MLP). See
[`bench/DESIGN.md`](bench/DESIGN.md) for the workload spec and parity
tolerances, and [`bench/results.md`](bench/results.md) for measured results
and correctness check output.

## Development

```bash
cargo test --lib        # 121 embedded parity tests
cargo build --examples  # ports of the Python examples/
```

Conversion rules live in [`CONVENTIONS.md`](CONVENTIONS.md).
