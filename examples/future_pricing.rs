//! Port of `../quantorch/examples/future_pricing.py`.
//!
//! Currency futures price via cost of carry (domestic rate minus foreign
//! rate), same parameters as the Python example.

fn main() {
    let domestic_rate = 0.05;
    let foreign_rate = 0.03;
    let spot = 1.0;
    let expiry = 1.0;

    let price = quantburn::future_pricer::future_pricer(spot, domestic_rate, expiry, Some(foreign_rate));
    println!("Currency Price: {price:.6}");
}
