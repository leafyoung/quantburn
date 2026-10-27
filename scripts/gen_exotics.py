"""Reference values for quantburn/src/exotics.rs (pure float64).

Monte-Carlo functions draw standard normals from scripts/refgen.py with the
SAME seeds used in the Rust tests, and consume them in the same order as the
Rust implementation (path-major, time-step inner; per-path asset normals for
the multi-asset pricers).
"""

import math
import sys

sys.path.insert(0, "scripts")
from refgen import normals


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cholesky(a):
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j]
            for k in range(j):
                s -= l[i][k] * l[j][k]
            l[i][j] = math.sqrt(s) if i == j else s / l[j][j]
    return l


def digital_option(option_type, spot, strike, expiry, volatility, rate, payout, num_paths, seed):
    z = normals(seed, num_paths)
    disc = math.exp(-rate * expiry)
    total = 0.0
    for zi in z:
        p = spot * math.exp((rate - 0.5 * volatility**2) * expiry + volatility * math.sqrt(expiry) * zi)
        hit = p >= strike if option_type == "call" else p <= strike
        total += (payout if hit else 0.0) * disc
    return total / num_paths


def rainbow_option(spots, weights, strike, expiry, vols, correlations, rate, num_paths, seed):
    n = len(spots)
    L = cholesky(correlations)
    z = normals(seed, num_paths * n)
    disc = math.exp(-rate * expiry)
    total = 0.0
    for path in range(num_paths):
        zv = z[path * n : (path + 1) * n]
        portfolio = 0.0
        for a in range(n):
            corr = sum(L[a][k] * zv[k] for k in range(n))
            p = spots[a] * math.exp(
                (rate - 0.5 * vols[a] ** 2) * expiry + vols[a] * math.sqrt(expiry) * corr
            )
            portfolio += p * weights[a]
        total += max(portfolio - strike, 0.0) * disc
    return total / num_paths


def _sim_paths(seed, num_paths, num_steps, spot, drift, sig_dt):
    z = normals(seed, num_paths * num_steps)
    paths = []
    for path in range(num_paths):
        price = spot
        row = [price]
        for t in range(num_steps):
            price *= math.exp(drift + sig_dt * z[path * num_steps + t])
            row.append(price)
        paths.append(row)
    return paths


def barrier_option(option_type, barrier_type, spot, strike, barrier, expiry, volatility, rate, num_paths, num_steps, seed):
    dt = expiry / num_steps
    drift = (rate - 0.5 * volatility**2) * dt
    sig_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)
    paths = _sim_paths(seed, num_paths, num_steps, spot, drift, sig_dt)
    total = 0.0
    for row in paths:
        if barrier_type == "up-and-out":
            hit = any(p > barrier for p in row)
        elif barrier_type == "down-and-out":
            hit = any(p < barrier for p in row)
        elif barrier_type == "up-and-in":
            hit = any(p >= barrier for p in row)
        else:
            hit = any(p <= barrier for p in row)
        payoff = max(row[-1] - strike, 0.0) if option_type == "call" else max(strike - row[-1], 0.0)
        if barrier_type.endswith("out"):
            payoff = 0.0 if hit else payoff
        else:
            payoff = payoff if hit else 0.0
        total += payoff * disc
    return total / num_paths


def lookback_option(option_type, spot, strike, expiry, volatility, rate, num_paths, strike_type, seed):
    dt = expiry / 252
    num_steps = int(expiry / dt)
    drift = (rate - 0.5 * volatility**2) * dt
    sig_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)
    paths = _sim_paths(seed, num_paths, num_steps, spot, drift, sig_dt)
    total = 0.0
    for row in paths:
        last, mx, mn = row[-1], max(row), min(row)
        if option_type == "call":
            payoff = max(mx - strike, 0.0) if strike_type == "fixed" else max(last - mx, 0.0)
        else:
            payoff = max(strike - mn, 0.0) if strike_type == "fixed" else max(mn - last, 0.0)
        total += payoff * disc
    return total / num_paths


def asian_option(option_type, spot, strike, expiry, volatility, rate, num_paths, num_steps, average_type, seed):
    dt = expiry / num_steps
    drift = (rate - 0.5 * volatility**2) * dt
    sig_dt = volatility * math.sqrt(dt)
    disc = math.exp(-rate * expiry)
    paths = _sim_paths(seed, num_paths, num_steps, spot, drift, sig_dt)
    total = 0.0
    for row in paths:
        if average_type == "arithmetic":
            avg = sum(row) / len(row)
        else:
            avg = math.exp(sum(math.log(p) for p in row) / len(row))
        payoff = max(avg - strike, 0.0) if option_type == "call" else max(strike - avg, 0.0)
        total += payoff * disc
    return total / num_paths


def quanto_option(spot, strike, expiry, volatility, fx_volatility, correlation, domestic_rate, foreign_rate, fx_rate, num_paths, seed):
    z = normals(seed, num_paths * 2)
    total = 0.0
    for path in range(num_paths):
        z1 = z[2 * path]
        z2 = correlation * z1 + math.sqrt(1 - correlation**2) * z[2 * path + 1]
        asset = spot * math.exp(
            (foreign_rate - 0.5 * volatility**2) * expiry + volatility * math.sqrt(expiry) * z1
        )
        fx = fx_rate * math.exp(
            (domestic_rate - foreign_rate - 0.5 * fx_volatility**2) * expiry
            + fx_volatility * math.sqrt(expiry) * z2
        )
        total += max(asset - strike, 0.0) * fx * math.exp(-domestic_rate * expiry)
    return total / num_paths


def basket_option(spots, weights, strike, expiry, vols, correlations, rate, option_type, num_paths, seed):
    n = len(spots)
    L = cholesky(correlations)
    z = normals(seed, num_paths * n)
    disc = math.exp(-rate * expiry)
    total = 0.0
    for path in range(num_paths):
        zv = z[path * n : (path + 1) * n]
        value = 0.0
        for a in range(n):
            corr = sum(L[a][k] * zv[k] for k in range(n))
            p = spots[a] * math.exp(
                (rate - 0.5 * vols[a] ** 2) * expiry + vols[a] * math.sqrt(expiry) * corr
            )
            value += p * weights[a]
        payoff = max(value - strike, 0.0) if option_type == "call" else max(strike - value, 0.0)
        total += payoff * disc
    return total / num_paths


def curran(option_type, average_type, spot, strike, expiry, volatility, rate, num_steps):
    adjusted_vol = volatility * math.sqrt((num_steps + 1) / (6 * num_steps))
    if average_type == "arithmetic":
        d1 = (math.log(spot / strike) + (rate + 0.5 * adjusted_vol**2) * expiry) / (adjusted_vol * math.sqrt(expiry))
        d2 = d1 - adjusted_vol * math.sqrt(expiry)
        if option_type == "call":
            return spot * math.exp(-rate * expiry) * normal_cdf(d1) - strike * math.exp(-rate * expiry) * normal_cdf(d2)
        return strike * math.exp(-rate * expiry) * normal_cdf(-d2) - spot * math.exp(-rate * expiry) * normal_cdf(-d1)
    adjusted_rate = rate - 0.5 * volatility**2 / 3
    # BSM european with dividend 0
    sq = math.sqrt(expiry)
    d1 = (math.log(spot / strike) + (adjusted_rate + 0.5 * adjusted_vol**2) * expiry) / (adjusted_vol * sq)
    d2 = d1 - adjusted_vol * sq
    if option_type == "call":
        return spot * normal_cdf(d1) - strike * math.exp(-adjusted_rate * expiry) * normal_cdf(d2)
    return strike * math.exp(-adjusted_rate * expiry) * normal_cdf(-d2) - spot * normal_cdf(-d1)


def chooser_option(spot, strike, expiry, volatility, rate, choose_time):
    t = expiry - choose_time
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * t) / (volatility * math.sqrt(t))
    d2 = d1 - volatility * math.sqrt(t)
    call = spot * normal_cdf(d1) - strike * math.exp(-rate * t) * normal_cdf(d2)
    put = strike * math.exp(-rate * t) * normal_cdf(-d2) - spot * normal_cdf(-d1)
    return max(call, put) * math.exp(-rate * choose_time)


if __name__ == "__main__":
    print("digital    :", repr(digital_option("call", 100, 100, 1, 0.2, 0.05, 10.0, 1000, 1234)))
    corr3 = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
    print("rainbow    :", repr(rainbow_option([100, 90, 80], [0.5, 0.3, 0.2], 90, 1, [0.2, 0.25, 0.3], corr3, 0.05, 500, 1235)))
    print("barrier uo :", repr(barrier_option("call", "up-and-out", 100, 100, 120, 1, 0.2, 0.05, 500, 50, 1236)))
    print("barrier di :", repr(barrier_option("put", "down-and-in", 100, 100, 85, 1, 0.2, 0.05, 500, 50, 1237)))
    print("lookback f :", repr(lookback_option("call", 100, 100, 0.5, 0.2, 0.05, 200, "fixed", 1238)))
    print("lookback fl:", repr(lookback_option("put", 100, 100, 0.5, 0.2, 0.05, 200, "floating", 1239)))
    print("asian a    :", repr(asian_option("call", 100, 100, 1, 0.2, 0.05, 1000, 50, "arithmetic", 1240)))
    print("asian g    :", repr(asian_option("put", 100, 100, 1, 0.2, 0.05, 1000, 50, "geometric", 1241)))
    print("quanto     :", repr(quanto_option(100, 100, 1, 0.2, 0.1, 0.3, 0.05, 0.02, 7.8, 1000, 1242)))
    corr2 = [[1.0, 0.4], [0.4, 1.0]]
    print("basket     :", repr(basket_option([100, 50], [0.6, 0.4], 75, 1, [0.2, 0.3], corr2, 0.05, "put", 1000, 1243)))
    print("curran a   :", repr(curran("call", "arithmetic", 100, 100, 1, 0.2, 0.05, 252)))
    print("curran g   :", repr(curran("put", "geometric", 100, 100, 1, 0.2, 0.05, 252)))
    print("chooser c  :", repr(chooser_option(100, 100, 1, 0.2, 0.05, 0.5)))
    print("chooser p  :", repr(chooser_option(80, 100, 1, 0.2, 0.05, 0.5)))
