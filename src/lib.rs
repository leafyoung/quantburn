//! quantburn — Rust/Burn port of torchquantlib (quantorch), one module per
//! source file, with embedded parity tests against Python reference values
//! (within floating-point tolerance).

pub mod rng;

// Bond & money markets
pub mod bond_pricer;
pub mod fixed_income_forward;
pub mod future_pricer;
pub mod letsberational;

// Options
pub mod american_option;
pub mod asian_option;
pub mod bermudan_option;
pub mod black_scholes_merton;
pub mod exotics;
pub mod option_utils;

// Models
pub mod binomial_tree_model;
pub mod stochastic_model;
pub mod black_karasinski;
pub mod cir;
pub mod hull_white;
pub mod lmm;
pub mod vasicek;
pub mod dupire_local_volatility;
pub mod gans;
pub mod heston;
pub mod sabr;

// Risk
pub mod greeks;
pub mod malliavin;
pub mod reduced_form_model;
pub mod structural_model;
pub mod expected_shortfall;
pub mod scenario_analysis;
pub mod stress_testing;
pub mod var;
pub mod valuation_adjustment;

// Utils
pub mod yield_curve_construction;
