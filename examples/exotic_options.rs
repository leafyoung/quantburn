//! Port of `../quantorch/examples/exotic_options.py`.
//!
//! Up-and-out barrier call priced by Monte Carlo with discrete barrier
//! monitoring, same option parameters as the Python example. The Rust API
//! additionally takes an explicit path count, step count and RNG seed (the
//! Python version used torch's global RNG); we use 100k paths, 100 steps and
//! the shared deterministic sampler with seed 42.

fn main() {
    let spot = 100.0;
    let strike = 105.0;
    let barrier = 110.0;
    let expiry = 1.0;
    let volatility = 0.2;
    let rate = 0.05;
    let steps = 100;

    let price = quantburn::exotics::barrier_option(
        "call",
        "up-and-out",
        spot,
        strike,
        barrier,
        expiry,
        volatility,
        rate,
        100_000,
        steps,
        42,
    );
    println!("Option Price: {price:.6}");
}
