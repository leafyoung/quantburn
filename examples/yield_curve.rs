//! Port of `../quantorch/examples/yield_curve.py`.
//!
//! Bootstrapped yields and a Nelson-Siegel curve. Adaptation: the Rust
//! `bootstrap_yield_curve` takes per-instrument cash-flow *vectors* (each
//! instrument discounts at exponent 1/(i+1)), so the Python example's scalar
//! cash flows are wrapped as one-element vectors, preserving the same
//! numbers and results.

fn main() {
    // Python: cash_flows = [1.0, 1.1, 1.2, 1.3, 1.4] (one scalar per
    // instrument). Rust API shape: one Vec<f64> per instrument.
    let cash_flows: Vec<Vec<f64>> = vec![
        vec![1.0],
        vec![1.1],
        vec![1.2],
        vec![1.3],
        vec![1.4],
    ];
    let prices = [0.95, 0.96, 0.97, 0.98, 0.99];

    let yields = quantburn::yield_curve_construction::bootstrap_yield_curve(&cash_flows, &prices)
        .expect("bootstrap");
    let ys: Vec<String> = yields.iter().map(|y| format!("{y:.6}")).collect();
    println!("Bootstrapped Yields: [{}]", ys.join(", "));

    let tau = [0.5, 1.0, 1.5, 2.0];
    let (beta0, beta1, beta2) = (0.03, -0.02, 0.02);
    let yields_ns = quantburn::yield_curve_construction::nelson_siegel_yield_curve(&tau, beta0, beta1, beta2);
    let ns: Vec<String> = yields_ns.iter().map(|y| format!("{y:.6}")).collect();
    println!("Nelson-Siegel Yields: [{}]", ns.join(", "));
}
