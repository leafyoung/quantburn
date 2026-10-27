"""Reference values for quantburn/src/bermudan_option.rs (pure float64)."""

import sys

sys.path.insert(0, "scripts")
from gen_option_utils import price_binomial_tree


def bermudan_option(option_type, spot, strike, expiry, volatility, rate, steps, exercise_dates):
    if volatility <= 0:
        raise RuntimeError("volatility must be positive for bermudan_option.")
    return price_binomial_tree(
        option_type, "bermudan", spot, strike, expiry, volatility, rate, steps,
        exercise_dates=exercise_dates,
    )


if __name__ == "__main__":
    print("put q  :", repr(bermudan_option("put", 100, 100, 1, 0.2, 0.05, 50, [12, 24, 36])))
    print("put [] :", repr(bermudan_option("put", 100, 100, 1, 0.2, 0.05, 50, [])))
    print("call   :", repr(bermudan_option("call", 100, 100, 1, 0.2, 0.05, 50, [10, 20, 30, 40])))
