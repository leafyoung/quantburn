//! Port of `../quantorch/examples/valuation_adjustment.py`.
//!
//! CVA / DVA / MVA / FVA valuation adjustments, same parameters as the
//! Python example.

fn main() {
    let exposure = 100.0;
    let default_prob = 0.02;
    let recovery_rate = 0.4;
    let funding_cost = 0.01;
    let funding_spread = 0.005;
    let maturity = 1.0;

    let cva = quantburn::valuation_adjustment::calculate_cva(exposure, default_prob, recovery_rate)
        .expect("CVA");
    let dva = quantburn::valuation_adjustment::calculate_dva(exposure, default_prob, recovery_rate)
        .expect("DVA");
    let mva = quantburn::valuation_adjustment::calculate_mva(exposure, funding_cost, maturity)
        .expect("MVA");
    let fva = quantburn::valuation_adjustment::calculate_fva(exposure, funding_spread, maturity)
        .expect("FVA");

    println!("CVA: {cva:.6}");
    println!("DVA: {dva:.6}");
    println!("MVA: {mva:.6}");
    println!("FVA: {fva:.6}");
}
