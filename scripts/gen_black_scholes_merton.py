"""Reference values for quantburn/src/black_scholes_merton.rs (pure float64)."""

import math


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def deterministic(option_type, spot, strike, expiry, rate, dividend):
    fwd = spot * math.exp((rate - dividend) * expiry)
    disc = math.exp(-rate * expiry)
    return disc * max(fwd - strike, 0.0) if option_type == "call" else disc * max(strike - fwd, 0.0)


def bsm(option_type, option_style, spot, strike, expiry, vol, rate, dividend):
    if vol <= 0:
        euro = deterministic(option_type, spot, strike, expiry, rate, dividend)
    else:
        sq = math.sqrt(expiry)
        d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol**2) * expiry) / (vol * sq)
        d2 = d1 - vol * sq
        nd1, nd2 = normal_cdf(d1), normal_cdf(d2)
        if option_type == "call":
            euro = spot * math.exp(-dividend * expiry) * nd1 - strike * math.exp(-rate * expiry) * nd2
        else:
            euro = strike * math.exp(-rate * expiry) * (1 - nd2) - spot * math.exp(-dividend * expiry) * (1 - nd1)
    if option_style == "european":
        return euro
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    return max(euro, intrinsic)


if __name__ == "__main__":
    print("euro call :", repr(bsm("call", "european", 100, 100, 1, 0.2, 0.05, 0.02)))
    print("euro put  :", repr(bsm("put", "european", 100, 100, 1, 0.2, 0.05, 0.02)))
    print("amer put  :", repr(bsm("put", "american", 80, 100, 1, 0.1, 0.05, 0.0)))
    print("amer call :", repr(bsm("call", "american", 90, 100, 1, 0.3, 0.02, 0.0)))
    print("det call  :", repr(bsm("call", "european", 100, 90, 2, 0.0, 0.03, 0.01)))
    print("det put   :", repr(bsm("put", "european", 100, 100, 2, 0.0, 0.03, 0.01)))
    # scipy cross-check of normal_cdf accuracy targets
    from scipy.stats import norm

    for x in (0.0, 1.0, -2.5, 6.0):
        print(f"ncdf({x}) :", repr(norm.cdf(x)))
