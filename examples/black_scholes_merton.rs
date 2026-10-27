//! Port of `../quantorch/examples/black_scholes_merton.py`.
//!
//! European call with a continuous dividend yield, closed form, same
//! parameters as the Python example.

fn main() {
    let spot = 100.0;
    let strike = 105.0;
    let expiry = 1.0;
    let volatility = 0.2;
    let rate = 0.05;
    let dividend = 0.02;

    let price = quantburn::black_scholes_merton::black_scholes_merton(
        "call", "european", spot, strike, expiry, volatility, rate, dividend,
    );
    println!("Option Price: {price:.6}");
}
