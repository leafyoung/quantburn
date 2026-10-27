"""Float64 reference values for fixed_income_forward, future_pricer,
letsberational, and yield_curve_construction parity tests (pure math)."""
import math


def fixed_income_forward(face_value, rate, time_to_maturity, forward_rate):
    return face_value * math.exp((forward_rate - rate) * time_to_maturity)


def future_pricer(spot, rate, expiry, foreign_rate=None):
    carry = rate if foreign_rate is None else rate - foreign_rate
    return spot * math.exp(carry * expiry)


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x):
    return 0.3989 * math.exp(-0.5 * x * x)


def black_scholes_price(vol, spot, strike, expiry, rate, is_call):
    sqrt_e = math.sqrt(expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * expiry) / (vol * sqrt_e)
    d2 = d1 - vol * sqrt_e
    df = math.exp(-rate * expiry)
    if is_call:
        return spot * normal_cdf(d1) - strike * df * normal_cdf(d2)
    return strike * df * normal_cdf(-d2) - spot * normal_cdf(-d1)


def implied_volatility(option_price, spot, strike, expiry, rate, is_call=True):
    df = math.exp(-rate * expiry)
    max_price = spot if is_call else strike * df
    if not (0 < option_price < max_price):
        raise RuntimeError("out of bounds")
    lower, upper = 1e-5, 5.0
    tol = 1e-8
    if option_price <= black_scholes_price(lower, spot, strike, expiry, rate, True):
        return lower
    for _ in range(100):
        mid = 0.5 * (lower + upper)
        diff = black_scholes_price(mid, spot, strike, expiry, rate, True) - option_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            upper = mid
        else:
            lower = mid
    raise RuntimeError("no converge")


def bootstrap_yield_curve(cash_flows, prices):
    def inner(flows, px):
        totals = [sum(row) for row in flows]
        n = len(totals)
        return [(totals[i] / px[i]) ** (1.0 / (i + 1)) - 1.0 for i in range(n)]
    if isinstance(cash_flows[0][0], list):
        return [inner(cf, p) for cf, p in zip(cash_flows, prices)]
    return inner(cash_flows, prices)


def nelson_siegel(tau, b0, b1, b2):
    out = []
    for t in tau:
        e = math.exp(-t)
        f1 = (1 - e) / t
        f2 = f1 - e
        out.append(b0 + b1 * f1 + b2 * f2)
    return out


if __name__ == "__main__":
    print("fwd        ", repr(fixed_income_forward(100.0, 0.05, 2.0, 0.03)))
    print("fut1       ", repr(future_pricer(100.0, 0.05, 1.0)))
    print("fut2       ", repr(future_pricer(100.0, 0.05, 0.5, foreign_rate=0.02)))
    print("ncdf       ", repr(normal_cdf(0.3)))
    print("npdf       ", repr(normal_pdf(0.3)))
    iv_call = implied_volatility(10.0, 100.0, 100.0, 1.0, 0.05, True)
    iv_put = implied_volatility(8.0, 100.0, 95.0, 0.5, 0.03, False)
    print("iv_call    ", repr(iv_call))
    print("bs_at_iv   ", repr(black_scholes_price(iv_call, 100.0, 100.0, 1.0, 0.05, True)))
    print("iv_put     ", repr(iv_put))
    print("boot2d     ", [repr(v) for v in bootstrap_yield_curve(
        [[50.0, 1050.0], [60.0, 1060.0], [1080.0, 0.0]], [980.0, 995.0, 900.0])])
    print("boot3d     ", [[repr(v) for v in batch] for batch in bootstrap_yield_curve(
        [[[50.0, 1050.0], [60.0, 1060.0]],
         [[40.0, 1040.0], [70.0, 1070.0]]],
        [[975.0, 990.0], [960.0, 1005.0]])])
    print("ns         ", [repr(v) for v in nelson_siegel(
        [0.5, 1.0, 2.0, 5.0], 0.03, -0.02, 0.01)])
