//! Port of `../quantorch/examples/binomial_tree.py`.
//!
//! European call priced on a CRR binomial tree, same parameters as the
//! Python example.

fn main() {
    let spot = 100.0;
    let strike = 105.0;
    let expiry = 1.0;
    let volatility = 0.2;
    let rate = 0.05;
    let steps = 100;

    let price = quantburn::binomial_tree_model::price_binomial_tree(
        "call", "european", spot, strike, expiry, volatility, rate, steps, None,
    );
    println!("Option Price: {price:.6}");
}
