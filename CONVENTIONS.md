# quantburn conversion conventions

Port of `../quantorch/torchquantlib` (Python/PyTorch) to Rust, one Rust module
per Python file. Crate root: `/var/home/yangye/devv/qfin/pricing/quantburn`.

Rules for every conversion:

1. **One Rust module per Python file.** The module file already exists as an
   empty stub under `src/` and is already declared in `src/lib.rs`. NEVER edit
   `src/lib.rs`, `src/rng.rs`, `Cargo.toml`, or anything outside your assigned
   files — other agents work in parallel.

2. **Read the Python source first.** Faithfully port the numerics (formulas,
   model dynamics, discretizations). Keep function names in snake_case close to
   the Python ones. Add a doc comment on the module naming the Python source
   file.

3. **Burn usage.** Use `burn::tensor::Tensor` with `burn_ndarray::NdArray<f32>`
   where tensor ops are natural (paths, grids, matmul). Plain `f64`/`Vec<f64>`
   is fine and preferred for closed-form scalar math — precision matters more
   than framework purity. Backend type: `type B = burn_ndarray::NdArray<f32>;`
   inside functions/tests, not a crate-wide export.

4. **Stochastic code.** Never use float RNG at runtime; draw standard normals
   from `crate::rng::Xorshift64Star` (already parity-verified against
   `scripts/refgen.py`, bit-exact). Pick a per-module seed (e.g. seed 1234 +
   module id) and document it.

5. **Embedded tests.** Every module gets `#[cfg(test)] mod tests` with:
   - deterministic closed-form results: expected values computed by a Python
     reference you run yourself (`python3`, torch 2.13 available), embedded as
     literals, tolerance `1e-8` relative (or absolute near 0);
   - stochastic results: write a script `scripts/gen_<module>.py` that
     re-implements the Python model in pure float64 (`math`/numpy float64),
     drawing normals from `scripts/refgen.py` with the SAME seed as the Rust
     code, and embed its outputs; tolerance `1e-6` relative.
   If the original torch code effectively computes in f32, widen tolerance to
   `1e-4` and note why.

6. **Validate.** Run `cargo test --lib <module_name>` until green. If Python
   and Rust disagree beyond tolerance, find and fix the discrepancy — do not
   loosen the tolerance without understanding why.

7. **Torch features with no Burn equivalent** (e.g. nn.Module training loops,
   optimizers): port the pricing/simulation core that the file contributes;
   if a file is purely a training harness with no reusable numeric core, put a
   short doc comment saying so and one small test proving the numeric pieces
   it does contain (if any). Report this in your final summary.

## Error-handling convention (going forward)

- `Result<_, String>` for **user-input validation** (mirrors the Python
  `ValueError`/`RuntimeError` raised on bad inputs).
- `panic!`/`assert!` only for **internal invariants or programmer errors**
  (broken preconditions that indicate a bug, not bad user data).
- The existing library largely follows the Python source semantics
  per-module: where the Python panics, the Rust panics; where it raises,
  the Rust returns `Err(String)`. Do not change a module's error style
  unless you also change the parity tests.

### Shared helpers — reuse, don't re-implement

- `crate::lmm::cholesky` — lower-triangular Cholesky with a
  positive-definiteness assert (used by lmm, exotics, greeks).
- `crate::black_scholes_merton::{erf, normal_cdf}` — high-accuracy
  (~1e-15) erf / standard normal CDF (used by black_scholes_merton,
  letsberational, exotics, ...).
- `crate::option_utils::{price_binomial_tree, binomial_tree_batch,
  calculate_binomial_tree_params}` — the single CRR binomial-tree core
  (used by option_utils consumers and `binomial_tree_model`).

### Documented port quirks (intentional — do not "fix" silently)

- `letsberational::implied_volatility` bisection always prices a CALL
  internally even for puts (`is_call` only gates the no-arbitrage upper
  bound) — a faithful port of the Python call-only pricing quirk.
- `greeks.rs`'s digital-option test uses epsilon = 2.0 because the Python
  default 1e-4 degenerates to zero (the finite-difference h is far too
  small for the payoff discontinuity — the Python code itself is weak
  there); the test documents the deviation.
