"""Reference values for quantburn/src/option_utils.rs (pure float64)."""

import math


def calculate_binomial_tree_params(expiry, volatility, rate, steps):
    dt = expiry / steps
    if dt <= 0:
        return dt, 1.0, 1.0, 1.0
    u = math.exp(volatility * math.sqrt(dt))
    d = 1 / u
    p = 1.0 if math.isclose(u, d) else (math.exp(rate * dt) - d) / (u - d)
    return dt, u, d, p


def option_payoff(option_type, prices, strike):
    if option_type == "call":
        return [max(s - strike, 0.0) for s in prices]
    if option_type == "put":
        return [max(strike - s, 0.0) for s in prices]
    raise ValueError(option_type)


def build_price_tree(spot, u, d, steps):
    tree = [[0.0] * (steps + 1) for _ in range(steps + 1)]
    for step in range(steps + 1):
        for down in range(step + 1):
            tree[step][down] = spot * u ** (step - down) * d ** down
    return tree


def price_binomial_tree(option_type, option_style, spot, strike, expiry, volatility, rate, steps, exercise_dates=None):
    if steps <= 0:
        raise ValueError
    if expiry <= 0:
        return option_payoff(option_type, [spot], strike)[0]
    if volatility < 0:
        raise RuntimeError
    if volatility == 0:
        terminal = spot * math.exp(rate * expiry)
        return math.exp(-rate * expiry) * option_payoff(option_type, [terminal], strike)[0]

    dt, u, d, p = calculate_binomial_tree_params(expiry, volatility, rate, steps)
    discount = math.exp(-rate * dt)
    price_tree = build_price_tree(spot, u, d, steps)
    value_tree = [[0.0] * (steps + 1) for _ in range(steps + 1)]
    value_tree[steps] = option_payoff(option_type, price_tree[steps], strike)

    if option_style == "american":
        exercise_steps = set(range(steps))
    elif option_style == "bermudan" and exercise_dates is not None:
        exercise_steps = {int(s) for s in exercise_dates if 0 <= int(s) < steps}
    else:
        exercise_steps = set()

    for step in range(steps - 1, -1, -1):
        for down in range(step + 1):
            continuation = (
                p * value_tree[step + 1][down] + (1 - p) * value_tree[step + 1][down + 1]
            ) * discount
            if step in exercise_steps:
                exercise = option_payoff(option_type, [price_tree[step][down]], strike)[0]
                value_tree[step][down] = max(exercise, continuation)
            else:
                value_tree[step][down] = continuation
    return value_tree[0][0]


if __name__ == "__main__":
    print("euro call 5   :", repr(price_binomial_tree("call", "european", 100, 100, 1, 0.2, 0.05, 5)))
    print("amer put 10   :", repr(price_binomial_tree("put", "american", 100, 100, 1, 0.2, 0.05, 10)))
    print("berm put 10   :", repr(price_binomial_tree("put", "bermudan", 100, 100, 1, 0.2, 0.05, 10, [2, 5, 8])))
    print("zero vol call :", repr(price_binomial_tree("call", "european", 100, 90, 1, 0.0, 0.05, 5)))
    print("zero expiry   :", repr(price_binomial_tree("put", "european", 95, 100, 0.0, 0.2, 0.05, 5)))
    print("params        :", tuple(repr(x) for x in calculate_binomial_tree_params(1.0, 0.2, 0.05, 4)))
