//! Port of `../quantorch/examples/credit_risk.py`.
//!
//! Merton structural-model equity value and reduced-form expected loss,
//! same parameters as the Python example.

fn main() {
    let asset_value = 100.0;
    let debt = 80.0;
    let volatility = 0.2;
    let rate = 0.05;
    let expiry = 1.0;

    let equity_value =
        quantburn::structural_model::merton_model(asset_value, debt, volatility, rate, expiry);
    println!("Equity Value (Merton Model): {equity_value:.6}");

    let lambda_0 = 0.02;
    let default_intensity = 0.05;
    let recovery_rate = 0.4;
    let time = 1.0;

    let expected_loss = quantburn::reduced_form_model::reduced_form_model(
        lambda_0,
        default_intensity,
        recovery_rate,
        time,
    );
    println!("Expected Loss (Reduced Form Model): {expected_loss:.6}");
}
