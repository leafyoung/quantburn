//! Port of `../quantorch/examples/malliavin_greeks.py`.
//!
//! Malliavin-calculus Greeks for a European call, same parameters
//! (S0=100, K=100, T=1, r=0.05, sigma=0.2) as the Python example. The Rust
//! API takes an explicit path count and seed (the Python version used
//! torch's global RNG); we use 1M paths with the shared deterministic
//! sampler, seed 42.

fn main() {
    let greeks = quantburn::greeks::european_option_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 1_000_000, 42);
    println!(
        "Delta: {:.6}\nGamma: {:.6}\nVega: {:.6}\nTheta: {:.6}\nRho: {:.6}",
        greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho
    );
}
