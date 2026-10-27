"""Reference values for quantburn/src/asian_option.rs (pure float64).

MC values replicate quantburn/src/exotics.rs asian_option draw order
(path-major, time-step inner) with refgen normals, seed 42.
"""

import math
import sys

sys.path.insert(0, "scripts")
from refgen import normals
from gen_exotics import asian_option as mc_asian

EPSILON = 1e-8


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sampling_times(expiry, steps):
    return [expiry * k / steps for k in range(1, steps + 1)]


def average_forward_mean(spot, expiry, rate, steps):
    times = sampling_times(expiry, steps)
    return spot * sum(math.exp(rate * t) for t in times) / steps


def discounted_intrinsic(option_type, average_level, strike, expiry, rate):
    payoff = max(average_level - strike, 0.0) if option_type == "call" else max(strike - average_level, 0.0)
    return math.exp(-rate * expiry) * payoff


def lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate):
    mean_level = max(mean_level, EPSILON)
    second_moment = max(second_moment, mean_level**2)
    log_variance = math.log(max(second_moment / max(mean_level**2, EPSILON), 1.0))
    if log_variance <= EPSILON:
        return discounted_intrinsic(option_type, mean_level, strike, expiry, rate)
    sqrt_lv = math.sqrt(log_variance)
    d1 = (math.log(mean_level / strike) + 0.5 * log_variance) / sqrt_lv
    d2 = d1 - sqrt_lv
    disc = math.exp(-rate * expiry)
    if option_type == "call":
        return disc * (mean_level * normal_cdf(d1) - strike * normal_cdf(d2))
    return disc * (strike * normal_cdf(-d2) - mean_level * normal_cdf(-d1))


def geometric_asian_price(option_type, spot, strike, expiry, volatility, rate, steps):
    times = sampling_times(expiry, steps)
    n = steps
    mean_t = sum(times) / n
    log_spot_mean = math.log(max(spot, EPSILON)) + (rate - 0.5 * volatility**2) * mean_t
    cov_mean = sum(min(ti, tj) for ti in times for tj in times) / (n * n)
    log_variance = volatility**2 * cov_mean
    second_moment = math.exp(2.0 * log_spot_mean + 2.0 * log_variance)
    mean_level = math.exp(log_spot_mean + 0.5 * log_variance)
    return lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)


def arithmetic_asian_lognormal_price(option_type, spot, strike, expiry, volatility, rate, steps):
    times = sampling_times(expiry, steps)
    n = steps
    mean_level = average_forward_mean(spot, expiry, rate, steps)
    second_moment = (
        spot**2
        * sum(
            math.exp(rate * (ti + tj) + volatility**2 * min(ti, tj))
            for ti in times
            for tj in times
        )
        / (n * n)
    )
    return lognormal_option_price(option_type, mean_level, second_moment, strike, expiry, rate)


class AsianOption:
    def __init__(self, underlying, strike, expiry, rf_rate, volatility, option_type="call", averaging_type="arithmetic", steps=252):
        self.u, self.k, self.t, self.r, self.v = underlying, strike, expiry, rf_rate, volatility
        self.option_type, self.averaging_type, self.steps = option_type, averaging_type, steps

    def analytical(self, steps=None):
        steps = steps or self.steps
        if self.v <= 0:
            return discounted_intrinsic(self.option_type, average_forward_mean(self.u, self.t, self.r, steps), self.k, self.t, self.r)
        if self.averaging_type == "geometric":
            return geometric_asian_price(self.option_type, self.u, self.k, self.t, self.v, self.r, steps)
        return arithmetic_asian_lognormal_price(self.option_type, self.u, self.k, self.t, self.v, self.r, steps)

    def monte_carlo(self, num_simulations, steps=None, seed=42):
        steps = steps or self.steps
        return mc_asian(self.option_type, self.u, self.k, self.t, self.v, self.r, num_simulations, steps, self.averaging_type, seed)

    def _bump(self, field, value):
        params = dict(u=self.u, k=self.k, t=self.t, r=self.r, v=self.v)
        params[field] = value
        return AsianOption(params["u"], params["k"], params["t"], params["r"], params["v"], self.option_type, self.averaging_type, self.steps).analytical()

    def delta(self):
        return (self._bump("u", self.u + 1e-2) - self._bump("u", self.u - 1e-2)) / 2e-2

    def gamma(self):
        return (self._bump("u", self.u + 1e-2) - 2.0 * self.analytical() + self._bump("u", self.u - 1e-2)) / 1e-4

    def theta(self):
        return -(self._bump("t", self.t + 1e-4) - self._bump("t", self.t - 1e-4)) / 2e-4

    def vega(self):
        return (self._bump("v", self.v + 1e-4) - self._bump("v", self.v - 1e-4)) / 2e-4

    def rho(self):
        return (self._bump("r", self.r + 1e-4) - self._bump("r", self.r - 1e-4)) / 2e-4


if __name__ == "__main__":
    geo = AsianOption(100, 100, 1, 0.05, 0.2, "call", "geometric", 252)
    ari = AsianOption(100, 100, 1, 0.05, 0.2, "call", "arithmetic", 252)
    put = AsianOption(100, 110, 1, 0.05, 0.2, "put", "arithmetic", 52)
    zerovol = AsianOption(100, 100, 1, 0.05, 0.0, "call", "arithmetic", 252)
    mc = AsianOption(100, 100, 1, 0.05, 0.2, "call", "arithmetic", 50)
    print("geo 252    :", repr(geo.analytical()))
    print("ari 252    :", repr(ari.analytical()))
    print("ari put 52 :", repr(put.analytical()))
    print("zero vol   :", repr(zerovol.analytical()))
    print("mc 42      :", repr(mc.monte_carlo(1000, 50, 42)))
    print("delta      :", repr(ari.delta()))
    print("gamma      :", repr(ari.gamma()))
    print("theta      :", repr(ari.theta()))
    print("vega       :", repr(ari.vega()))
    print("rho        :", repr(ari.rho()))
