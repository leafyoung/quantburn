//! Port of `../quantorch/examples/bayesian_inference.py`.
//!
//! The Python example is entirely example-local: it defines a discrete
//! Bayesian update (prior * likelihood, renormalized) on plain tensors and
//! uses no torchquantlib functionality. Reproduced verbatim here in Rust
//! with f64 slices; no library call is involved.

/// Discrete Bayes update: elementwise prior * likelihood, then normalize.
fn bayesian_update(prior: &[f64], likelihood: &[f64]) -> Vec<f64> {
    assert_eq!(prior.len(), likelihood.len(), "shape mismatch");
    let mut posterior: Vec<f64> = prior
        .iter()
        .zip(likelihood)
        .map(|(&p, &l)| p * l)
        .collect();
    let total: f64 = posterior.iter().sum();
    for v in &mut posterior {
        *v /= total;
    }
    posterior
}

fn main() {
    let prior = [0.2, 0.3, 0.5];
    let likelihood = [0.6, 0.1, 0.3];
    let posterior = bayesian_update(&prior, &likelihood);
    let ps: Vec<String> = posterior.iter().map(|p| format!("{p:.6}")).collect();
    println!("Posterior: [{}]", ps.join(", "));
}
