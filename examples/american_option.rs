//! Port of `../quantorch/examples/american_option.py`.
//!
//! Prices an American call with a CRR binomial tree using the same
//! parameters as the Python example.

fn main() {
    let spot = 100.0;
    let strike = 105.0;
    let expiry = 1.0;
    let volatility = 0.2;
    let rate = 0.05;
    let steps = 100;

    let price = quantburn::american_option::american_option("call", spot, strike, expiry, volatility, rate, steps);
    println!("Option Price: {price:.6}");
}
