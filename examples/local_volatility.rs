//! Port of `../quantorch/examples/local_volatility.py`.
//!
//! Dupire local-volatility simulation with a constant 20% surface, same
//! parameters (S0=100, T=1, N=1000, rate=0.05, steps=252) as the Python
//! example. Normals come from the shared deterministic sampler; the parity
//! tests use seed 2009, which we reuse here so the first terminal prices are
//! reproducible.

fn main() {
    let local_vol_func = |_spot: f64, _time: f64| 0.2;

    let terminal_prices =
        quantburn::dupire_local_volatility::simulate(&local_vol_func, 100.0, 1.0, 1000, 252, 0.05, 2009);
    let head: Vec<String> = terminal_prices[..5]
        .iter()
        .map(|p| format!("{p:.6}"))
        .collect();
    println!("Terminal Prices: [{}]", head.join(", "));
    let mean: f64 = terminal_prices.iter().sum::<f64>() / terminal_prices.len() as f64;
    println!("Mean Terminal Price: {mean:.6}");
}
