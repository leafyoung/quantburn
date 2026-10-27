//! Adapted from `../quantorch/examples/calibration.py`.
//!
//! The Python example is a PyTorch optimizer loop (`ModelCalibrator`
//! gradient descent on Heston parameters against simulated paths). quantburn
//! does not port training loops — quantorch's README itself marks the
//! calibration harness as "no portable numeric core" — so this example
//! demonstrates the portable piece the crate *does* expose:
//! [`quantburn::stochastic_model::StochasticModel`], the parameter store
//! with box-constraint handling that underpins calibration. It builds the
//! same true vs. initial-guess Heston parameter sets, registers the same
//! bounded parameters, and shows constraint clipping plus the simulated-path
//! entry point (`heston::simulate`) that a calibration loop would drive.

use quantburn::stochastic_model::{ParamBounds, StochasticModel};

fn main() {
    // True parameters (as in the Python `heston_true`) and initial guesses
    // (`heston_model`).
    let true_params = quantburn::heston::HestonParams {
        kappa: 2.0,
        theta: 0.04,
        sigma_v: 0.3,
        rho: -0.7,
        v0: 0.04,
        mu: 0.05,
    };
    println!("True parameters: {true_params:?}");

    // Parameter store with calibration bounds (positive mean reversion and
    // vol-of-vol, rho in [-1, 1]).
    let mut model = StochasticModel::new();
    model.set_param("kappa", 1.0);
    model.set_param("theta", 0.02);
    model.set_param("sigma_v", 0.2);
    model.set_param("rho", -0.5);
    model.set_param("v0", 0.02);
    model.set_param("mu", 0.0);
    model.set_constraints("kappa", ParamBounds::new(Some(1e-8), None));
    model.set_constraints("theta", ParamBounds::new(Some(1e-8), None));
    model.set_constraints("sigma_v", ParamBounds::new(Some(1e-8), None));
    model.set_constraints("rho", ParamBounds::new(Some(-1.0), Some(1.0)));
    model.set_constraints("v0", ParamBounds::new(Some(1e-8), None));

    // Constraint handling: raw (possibly out-of-bounds) values are clipped
    // to the registered bounds when read back.
    model.set_param("kappa", -0.5);
    println!(
        "kappa raw -0.5, constrained: {}",
        model.constrained_param("kappa")
    );
    model.apply_constraints();
    model.set_param("kappa", 1.0);

    let initial: Vec<(String, f64)> = ["kappa", "theta", "sigma_v", "rho", "v0", "mu"]
        .iter()
        .map(|&n| (n.to_string(), model.param(n).unwrap_or(f64::NAN)))
        .collect();
    println!("Initial guess parameters: {initial:?}");

    // Simulated observed paths are what a calibration loop would score the
    // model against; here we generate them with the true parameters using
    // the deterministic sampler (S0=100, T=1, N=1000, steps=100, seed 42).
    let observed = quantburn::heston::simulate(true_params, 100.0, 1.0, 1000, 100, 42);
    let mean: f64 = observed.iter().sum::<f64>() / observed.len() as f64;
    println!(
        "Observed terminal prices: N={} mean={mean:.6} (training loop not ported; see src/stochastic_model.rs)",
        observed.len()
    );
}
