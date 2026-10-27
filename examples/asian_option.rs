//! Port of `../quantorch/examples/asian_option.py`.
//!
//! Prices an arithmetic-average Asian call. The Python wrapper defaults to
//! `average_type="arithmetic"`; the Rust [`quantburn::asian_option`] takes it
//! explicitly. With `num_paths = None` the closed-form (Levy-style analytic
//! approximation) price is returned, matching the Python default path.

fn main() {
    let spot = 100.0;
    let strike = 105.0;
    let expiry = 1.0;
    let volatility = 0.2;
    let rate = 0.05;
    let steps = 100;

    let price = quantburn::asian_option::asian_option(
        "call", spot, strike, expiry, volatility, rate, steps, "arithmetic", None, 42,
    );
    println!("Option Price: {price:.6}");

    // Monte Carlo variant (shared deterministic RNG, seed 42).
    let mc_price = quantburn::asian_option::asian_option(
        "call", spot, strike, expiry, volatility, rate, steps, "arithmetic", Some(100_000), 42,
    );
    println!("Option Price (Monte Carlo, 100k paths): {mc_price:.6}");
}
