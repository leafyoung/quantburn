//! Port of `../quantorch/examples/vasicek.py`.
//!
//! Vasicek short-rate simulation: N draws from the exact terminal
//! distribution r(T), same model parameters (kappa=0.5, theta=0.04,
//! sigma=0.01, r0=0.03, T=1, N=1000) as the Python example. The Python
//! version relied on torch's global RNG; the Rust port uses the shared
//! deterministic sampler (seed 2002, as in the parity tests).

fn main() {
    let (kappa, theta, sigma, r0) = (0.5, 0.04, 0.01, 0.03);
    let rates = quantburn::vasicek::simulate(kappa, theta, sigma, r0, 1.0, 1000, 2002);

    let head: Vec<String> = rates[..5].iter().map(|r| format!("{r:.6}")).collect();
    println!("Interest Rates (first 5 of {}): [{}]", rates.len(), head.join(", "));
    let mean: f64 = rates.iter().sum::<f64>() / rates.len() as f64;
    let min = rates.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Mean: {mean:.6}  Min: {min:.6}  Max: {max:.6}");

    // Closed-form zero-coupon bond price/yield from the same model.
    let zcb = quantburn::vasicek::zero_coupon_bond_price(kappa, theta, sigma, r0, 1.0);
    let zcy = quantburn::vasicek::zero_coupon_yield(kappa, theta, sigma, r0, 1.0);
    println!("Vasicek 1y ZCB price: {zcb:.6}  yield: {zcy:.6}");
}
