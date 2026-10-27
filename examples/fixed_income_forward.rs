//! Port of `../quantorch/examples/fixed_income_forward.py`.
//!
//! Forward price of a fixed income security under continuous compounding,
//! same parameters as the Python example.

fn main() {
    let face_value = 1000.0;
    let rate = 0.05;
    let time_to_maturity = 1.0;
    let forward_rate = 0.06;

    let price =
        quantburn::fixed_income_forward::fixed_income_forward(face_value, rate, time_to_maturity, forward_rate);
    println!("Fixed Income Forward Price: {price:.6}");
}
