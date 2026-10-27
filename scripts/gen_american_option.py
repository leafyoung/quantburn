"""Reference values for quantburn/src/american_option.rs (pure float64)."""

import sys

sys.path.insert(0, "scripts")
from gen_option_utils import price_binomial_tree


def american_option(option_type, spot, strike, expiry, volatility, rate, steps):
    if volatility <= 0:
        raise RuntimeError("volatility must be positive for american_option.")
    return price_binomial_tree(option_type, "american", spot, strike, expiry, volatility, rate, steps)


if __name__ == "__main__":
    print("put 50 :", repr(american_option("put", 100, 100, 1, 0.2, 0.05, 50)))
    print("call 50:", repr(american_option("call", 100, 100, 1, 0.2, 0.05, 50)))
    print("itm put:", repr(american_option("put", 80, 100, 1, 0.2, 0.05, 10)))
