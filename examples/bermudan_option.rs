//! Port of `../quantorch/examples/bermudan_option.py`.
//!
//! Prices a Bermudan call with exercise allowed at steps 30, 60 and 90 of a
//! 100-step tree, same parameters as the Python example.

fn main() {
    let spot = 100.0;
    let strike = 105.0;
    let expiry = 1.0;
    let volatility = 0.2;
    let rate = 0.05;
    let steps = 100;
    let exercise_dates = [30usize, 60, 90];

    let price = quantburn::bermudan_option::bermudan_option(
        "call",
        spot,
        strike,
        expiry,
        volatility,
        rate,
        steps,
        &exercise_dates,
    );
    println!("Option Price: {price:.6}");
}
