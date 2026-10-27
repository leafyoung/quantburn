//! Port of `../quantorch/examples/heston_option_pricing.py`.
//!
//! Heston European call by Monte Carlo (antithetic Euler with martingale
//! normalization), same model parameters, S0, K, T, r and simulation size
//! (N=100000, steps=200) as the Python example. The Python version relied on
//! torch's global RNG with a hard-coded internal seed; the Rust port takes
//! the seed explicitly (deterministic shared sampler, seed 2002 as used in
//! the parity tests).

fn main() {
    let params = quantburn::heston::HestonParams {
        kappa: 2.0,
        theta: 0.04,
        sigma_v: 0.3,
        rho: -0.7,
        v0: 0.04,
        mu: 0.05,
    };

    let s0 = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.05;
    let option_type = "call";

    let option_price = quantburn::heston::monte_carlo_option_price(
        params, s0, k, t, r, option_type, 0.0, 100_000, 200, 2002,
    );
    println!("The {option_type} option price under the Heston model is: {option_price:.4}");

    // Closed-form reference via the characteristic-function (Carr-Madan)
    // pricer, for comparison.
    let cf_price = quantburn::heston::characteristic_function_price(
        params, s0, k, t, r, option_type, 0.0, 1000.0, 100_000,
    );
    println!("Characteristic-function price: {cf_price:.4}");
}
