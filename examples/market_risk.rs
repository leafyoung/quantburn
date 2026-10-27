//! Port of `../quantorch/examples/market_risk.py`.
//!
//! Historical VaR and Expected Shortfall on a small return series at the
//! 95% confidence level, same data as the Python example.

fn main() {
    let returns = [0.01, -0.02, 0.03, -0.01, 0.04, -0.03];
    let confidence_level = 0.95;

    let var = quantburn::var::calculate_var(&returns, confidence_level).expect("VaR");
    let es = quantburn::expected_shortfall::calculate_es(&returns, confidence_level)
        .expect("Expected Shortfall");

    println!("VaR: {var:.6}");
    println!("Expected Shortfall: {es:.6}");
}
