//! Port of `torchquantlib/models/stochastic_volatility/heston.py`
//! (Heston stochastic volatility model).
//!
//! - `simulate` / `simulate_terminal_paths`: full-truncation Euler
//!   discretization of the Heston dynamics, terminal prices only;
//! - `characteristic_function_price`: Lewis-style integration of the
//!   log-price characteristic function with the trapezoid rule;
//! - `monte_carlo_option_price`: antithetic MC with martingale
//!   normalization.
//!
//! Torch nn.Module training machinery has no counterpart here (rule 7).
//! Module RNG seed: 2007 (Heston) — normals come from
//! `crate::rng::Xorshift64Star`, drawn per step as a block of `base_count`
//! spot normals followed by a block of `base_count` auxiliary normals,
//! matching the Python `_draw_normals` order.

/// Heston model parameters (already-constrained values as the Python
/// `constrained_param` would return them).
#[derive(Debug, Clone, Copy)]
pub struct HestonParams {
    pub kappa: f64,
    pub theta: f64,
    pub sigma_v: f64,
    pub rho: f64,
    pub v0: f64,
    pub mu: f64,
}

impl Default for HestonParams {
    fn default() -> Self {
        Self { kappa: 1.0, theta: 0.04, sigma_v: 0.5, rho: -0.5, v0: 0.04, mu: 0.0 }
    }
}

const EPSILON: f64 = 1e-8;

/// Minimal f64 complex number (no num-complex dependency available).
#[derive(Debug, Clone, Copy)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn sub(self, o: Self) -> Self {
        Self { re: self.re - o.re, im: self.im - o.im }
    }
    fn add(self, o: Self) -> Self {
        Self { re: self.re + o.re, im: self.im + o.im }
    }
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self {
            re: (self.re * o.re + self.im * o.im) / d,
            im: (self.im * o.re - self.re * o.im) / d,
        }
    }
    fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
    /// Principal square root.
    fn sqrt(self) -> Self {
        let r = self.re.hypot(self.im);
        let m = r.sqrt();
        let th = self.im.atan2(self.re) / 2.0;
        Self { re: m * th.cos(), im: m * th.sin() }
    }
    /// Principal natural logarithm.
    fn ln(self) -> Self {
        Self { re: self.re.hypot(self.im).ln(), im: self.im.atan2(self.re) }
    }
    fn exp(self) -> Self {
        let e = self.re.exp();
        Self { re: e * self.im.cos(), im: e * self.im.sin() }
    }
    fn powi2(self) -> Self {
        self.mul(self)
    }
}

/// Python `_simulate_terminal_paths`. When `seed` is Some, normals are drawn
/// from `crate::rng::Xorshift64Star::new(seed)`; when None, the test-only
/// convention is seed 2007.
pub fn simulate_terminal_paths(
    params: HestonParams,
    s0: f64,
    t: f64,
    n: usize,
    steps: usize,
    drift: Option<f64>,
    antithetic: bool,
    normalize_martingale: bool,
    seed: u64,
) -> Vec<f64> {
    let spot = s0.max(EPSILON);
    if t <= 0.0 {
        return vec![spot; n];
    }
    let steps = steps.max(1);
    let dt = t / steps as f64;
    let sqrt_dt = dt.sqrt();

    let model_drift = drift.unwrap_or(params.mu);
    let base_count = if antithetic { (n + 1) / 2 } else { n };

    let rho_residual = (1.0 - params.rho * params.rho).clamp(EPSILON, f64::MAX).sqrt();

    let mut rng = crate::rng::Xorshift64Star::new(seed);

    // Antithetic padding mirrors `tensor.repeat(2)[:num_paths]`; since the
    // initial arrays are constant, a full-length constant vector is identical.
    let mut prices: Vec<f64> = vec![spot; n];
    let mut variance: Vec<f64> = vec![params.v0; n];

    for _ in 0..steps {
        let z_spot_raw: Vec<f64> = (0..base_count).map(|_| rng.next_std_normal()).collect();
        let z_aux_raw: Vec<f64> = (0..base_count).map(|_| rng.next_std_normal()).collect();
        let z_spot = if antithetic {
            let mut z = z_spot_raw.clone();
            z.extend(z_spot_raw.iter().map(|v| -v));
            z.truncate(n);
            z
        } else {
            z_spot_raw
        };
        let z_aux = if antithetic {
            let mut z = z_aux_raw.clone();
            z.extend(z_aux_raw.iter().map(|v| -v));
            z.truncate(n);
            z
        } else {
            z_aux_raw
        };

        for i in 0..n {
            let z_var = params.rho * z_spot[i] + rho_residual * z_aux[i];
            let variance_pos = variance[i].max(0.0);
            let new_variance = (variance[i]
                + params.kappa * (params.theta - variance_pos) * dt
                + params.sigma_v * variance_pos.max(EPSILON).sqrt() * sqrt_dt * z_var)
                .max(0.0);
            prices[i] *= ((model_drift - 0.5 * variance_pos) * dt
                + variance_pos.max(EPSILON).sqrt() * sqrt_dt * z_spot[i])
            .exp();
            variance[i] = new_variance;
        }
    }

    if normalize_martingale {
        let target_forward = spot * (model_drift * t).exp();
        let mean: f64 = prices.iter().sum::<f64>() / prices.len() as f64;
        let scale = target_forward / mean.max(EPSILON);
        for p in &mut prices {
            *p *= scale;
        }
    }
    prices
}

/// Python `Heston.simulate` (plain Euler, no antithetic / normalization).
pub fn simulate(params: HestonParams, s0: f64, t: f64, n: usize, steps: usize, seed: u64) -> Vec<f64> {
    simulate_terminal_paths(params, s0, t, n, steps, None, false, false, seed)
}

/// Python `_log_price_characteristic_function` at complex argument `u`.
fn log_price_characteristic_function(
    params: HestonParams,
    u: C64,
    spot: f64,
    expiry: f64,
    rate: f64,
    dividend: f64,
) -> C64 {
    let kappa = params.kappa;
    let theta = params.theta;
    let sigma_v = params.sigma_v;
    let rho = params.rho;
    let v0 = params.v0;

    let a = kappa * theta;
    let i = C64::new(0.0, 1.0);
    let iu = i.mul(u);
    let d = (iu.scale(rho * sigma_v).sub(C64::new(kappa, 0.0))).powi2()
        .add(C64::new(sigma_v * sigma_v, 0.0).mul(u.powi2().add(iu)))
        .sqrt();
    let g = (C64::new(kappa, 0.0).sub(iu.scale(rho * sigma_v)).sub(d))
        .div(C64::new(kappa, 0.0).sub(iu.scale(rho * sigma_v)).add(d));
    let exp_neg_d_t = d.scale(-expiry).exp();

    let c_term = C64::new((rate - dividend) * expiry, 0.0).mul(iu)
        .add(C64::new(a / (sigma_v * sigma_v), 0.0).mul(
            C64::new(kappa, 0.0).sub(iu.scale(rho * sigma_v)).sub(d).scale(expiry)
                .sub((C64::new(1.0, 0.0).sub(g.mul(exp_neg_d_t))).div(C64::new(1.0, 0.0).sub(g)).ln().scale(2.0)),
        ));
    let d_term = C64::new(kappa, 0.0).sub(iu.scale(rho * sigma_v)).sub(d)
        .scale(1.0 / (sigma_v * sigma_v))
        .mul(C64::new(1.0, 0.0).sub(exp_neg_d_t)
            .div(C64::new(1.0, 0.0).sub(g.mul(exp_neg_d_t))));
    c_term.add(d_term.scale(v0)).add(iu.scale(spot.ln())).exp()
}

/// Python `characteristic_function_price` (default method): trapezoid
/// integration over u in [1e-5, integration_limit] with `integration_points`
/// points (at least 256).
pub fn characteristic_function_price(
    params: HestonParams,
    s0: f64,
    k: f64,
    t: f64,
    r: f64,
    option_type: &str,
    q: f64,
    integration_limit: f64,
    integration_points: usize,
) -> f64 {
    let option_style = option_type.to_ascii_lowercase();
    assert!(option_style == "call" || option_style == "put", "Invalid option type. Use 'call' or 'put'.");
    let spot = s0.max(EPSILON);
    let strike = k.max(EPSILON);

    if t <= 0.0 {
        return if option_style == "call" {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
    }

    let n = integration_points.max(256);
    let u_lo = 1e-5_f64;
    let du = (integration_limit - u_lo) / (n - 1) as f64;
    let log_strike = strike.ln();
    let i = C64::new(0.0, 1.0);
    let phi_minus_i = log_price_characteristic_function(params, C64::new(0.0, -1.0), spot, t, r, q);

    let mut p1_int = 0.0;
    let mut p2_int = 0.0;
    for idx in 0..n {
        let u = u_lo + du * idx as f64;
        let kernel = i.scale(-u * log_strike).exp().div(i.scale(u));
        let p1_i = kernel.mul(log_price_characteristic_function(params, C64::new(u, -1.0), spot, t, r, q))
            .div(phi_minus_i)
            .re;
        let p2_i = kernel.mul(log_price_characteristic_function(params, C64::new(u, 0.0), spot, t, r, q)).re;
        let w = if idx == 0 || idx == n - 1 { 0.5 } else { 1.0 };
        p1_int += w * p1_i * du;
        p2_int += w * p2_i * du;
    }
    let p1 = 0.5 + p1_int / std::f64::consts::PI;
    let p2 = 0.5 + p2_int / std::f64::consts::PI;
    let discount_dividend = (-q * t).exp();
    let discount_rate = (-r * t).exp();
    let call_price = spot * discount_dividend * p1 - strike * discount_rate * p2;

    if option_style == "call" {
        call_price.max(0.0)
    } else {
        (call_price - spot * discount_dividend + strike * discount_rate).max(0.0)
    }
}

/// Python `monte_carlo_option_price`: antithetic Euler with martingale
/// normalization and deterministic seed.
pub fn monte_carlo_option_price(
    params: HestonParams,
    s0: f64,
    k: f64,
    t: f64,
    r: f64,
    option_type: &str,
    q: f64,
    n: usize,
    steps: usize,
    seed: u64,
) -> f64 {
    let option_style = option_type.to_ascii_lowercase();
    assert!(option_style == "call" || option_style == "put", "Invalid option type. Use 'call' or 'put'.");
    let terminal_prices = simulate_terminal_paths(
        params,
        s0,
        t,
        n,
        steps,
        Some(r - q),
        true,
        true,
        seed,
    );
    let payoff: f64 = terminal_prices
        .iter()
        .map(|s| {
            if option_style == "call" {
                (s - k).max(0.0)
            } else {
                (k - s).max(0.0)
            }
        })
        .sum();
    (-r * t).exp() * payoff / terminal_prices.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> HestonParams {
        HestonParams::default()
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Characteristic-function price; reference from
    /// scripts/gen_heston.py (pure float64 replication of the torch code),
    /// tolerance 1e-10 relative.
    #[test]
    fn cf_price_matches_reference() {
        let p = params();
        let call = characteristic_function_price(p, 100.0, 100.0, 1.0, 0.05, "call", 0.0, 100.0, 2000);
        let put = characteristic_function_price(p, 100.0, 100.0, 1.0, 0.05, "put", 0.0, 100.0, 2000);
        let call_otm = characteristic_function_price(p, 100.0, 120.0, 0.5, 0.03, "call", 0.01, 100.0, 2000);
        assert!(rel(call, 9.89702247730795) < 1e-10, "call={call}");
        assert!(rel(put, 5.0199649273793625) < 1e-10, "put={put}");
        assert!(rel(call_otm, 0.3987289754945964) < 1e-10, "call_otm={call_otm}");
        // Zero expiry -> intrinsic.
        let intr = characteristic_function_price(p, 110.0, 100.0, 0.0, 0.05, "call", 0.0, 100.0, 256);
        assert!((intr - 10.0).abs() < 1e-12);
    }

    /// Stochastic terminal prices via `simulate`; expected values from
    /// scripts/gen_heston.py with refgen normals, seed 2007, tol 1e-12.
    #[test]
    fn simulate_matches_reference() {
        let out = simulate(params(), 100.0, 1.0, 6, 32, 2007);
        let expected = [
            65.87079325186761,
            78.20202889639539,
            119.95791378057932,
            110.61440405377486,
            87.21994074853231,
            128.78429604873406,
        ];
        for (got, want) in out.iter().zip(expected) {
            assert!(rel(*got, want) < 1e-12, "got={got} want={want}");
        }
    }

    /// Antithetic MC with martingale normalization; expected values from
    /// scripts/gen_heston.py, seed 2007, tol 1e-12 on terminal paths, 1e-9
    /// on the averaged price.
    #[test]
    fn monte_carlo_price_matches_reference() {
        let price = monte_carlo_option_price(params(), 100.0, 100.0, 1.0, 0.05, "call", 0.0, 64, 32, 2007);
        assert!(rel(price, 9.586220714087805) < 1e-9, "price={price}");
        let put = monte_carlo_option_price(params(), 100.0, 100.0, 1.0, 0.05, "put", 0.0, 64, 32, 2007);
        assert!(rel(put, 4.709163164159194) < 1e-9, "put={put}");
    }
}
