//! Port of `../quantorch/examples/bond_pricing.py`.
//!
//! Zero-coupon, coupon, stochastic-rate, callable and convertible bond
//! pricing with the same parameters as the Python example.

fn main() {
    let face_value = 1000.0;
    let rate = 0.05;
    let maturity = 5.0;
    let coupon_rate = 0.06;
    let periods = 10i64;

    let zcb_price = quantburn::bond_pricer::zero_coupon_bond(face_value, rate, maturity);
    println!("Zero Coupon Bond Price: {zcb_price:.6}");

    let cb_price = quantburn::bond_pricer::coupon_bond(face_value, coupon_rate, rate, periods);
    println!("Coupon Bond Price: {cb_price:.6}");

    let rate_tensor = [
        0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095,
    ];
    let srb_price =
        quantburn::bond_pricer::stochastic_rate_bond(face_value, coupon_rate, &rate_tensor);
    println!("Stochastic Rate Bond Price: {srb_price:.6}");

    let call_price = 1050.0;
    let call_period = 5i64;
    let callable_price = quantburn::bond_pricer::callable_bond(
        face_value,
        coupon_rate,
        rate,
        periods,
        call_price,
        call_period,
    );
    println!("Callable Bond Price: {callable_price:.6}");

    let conversion_ratio = 1.1;
    let conversion_price = 100.0;
    let convertible_price = quantburn::bond_pricer::convertible_bond(
        face_value,
        coupon_rate,
        rate,
        periods,
        conversion_ratio,
        conversion_price,
    );
    println!("Convertible Bond Price: {convertible_price:.6}");
}
