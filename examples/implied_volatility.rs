//! Port of `../quantorch/examples/implied_volatility.py`.
//!
//! "Let's be rational"-style implied volatility solved here with the ported
//! robust bisection search, same market quote as the Python example.

fn main() {
    let market_price = 10.0;
    let spot = 100.0;
    let strike = 105.0;
    let expiry = 1.0;
    let rate = 0.05;

    let iv = quantburn::letsberational::implied_volatility(market_price, spot, strike, expiry, rate, true)
        .expect("implied volatility solve");
    println!("Implied Volatility: {iv:.6}");
}
