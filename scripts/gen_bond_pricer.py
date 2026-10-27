"""Float64 reference values for bond_pricer parity tests (pure math, no torch).

Mirrors torchquantlib/core/asset_pricing/bond/bond_pricer.py numerics.
"""
import math

EPS = 1e-8


def short_rate_tree(initial_rate, periods, sigma):
    n = max(1, int(periods))
    sigma = max(sigma, 0.0)
    if sigma == 0:
        return [[initial_rate] * (k + 1) for k in range(n)]
    up = math.exp(sigma)
    tree = []
    for k in range(n):
        exponents = list(range(-k, k + 1, 2))
        tree.append([max(initial_rate * up ** e, 0.0) for e in exponents])
    return tree


def embedded_option_bond_lattice(face, coupon_rate, rate, periods,
                                 exercise_price=None, exercise_period=None,
                                 exercise_kind=None, rate_volatility=None):
    n = max(1, int(periods))
    rate_tree = short_rate_tree(rate, n, rate_volatility or 0.0)
    coupon = face * coupon_rate
    continuation = [face] * (n + 1)
    ex_idx = None if exercise_period is None else max(1, min(int(exercise_period), n))
    for step in range(n - 1, -1, -1):
        expected_next = [0.5 * (continuation[i] + continuation[i + 1]) for i in range(len(continuation) - 1)]
        node_values = [(coupon + expected_next[i]) / max(1.0 + rate_tree[step][i], EPS)
                       for i in range(len(expected_next))]
        if ex_idx is not None and exercise_price is not None and step + 1 == ex_idx:
            if exercise_kind == "call":
                node_values = [min(v, exercise_price) for v in node_values]
            elif exercise_kind == "put":
                node_values = [max(v, exercise_price) for v in node_values]
        continuation = node_values
    return continuation[0]


def zero_coupon_bond(face, rate, maturity):
    return face / max((1 + rate) ** maturity, EPS)


def coupon_bond(face, coupon_rate, rate, periods):
    n = max(1, int(periods))
    cf = face * coupon_rate
    s = sum(cf / max((1 + rate) ** t, EPS) for t in range(1, n + 1))
    return s + face / max((1 + rate) ** n, EPS)


def callable_bond_lattice(face, cr, rate, periods, call_price, call_period, vol=None):
    return min(embedded_option_bond_lattice(face, cr, rate, periods, call_price, call_period, "call", vol),
               coupon_bond(face, cr, rate, periods))


def putable_bond_lattice(face, cr, rate, periods, put_price, put_period, vol=None):
    return max(embedded_option_bond_lattice(face, cr, rate, periods, put_price, put_period, "put", vol),
               coupon_bond(face, cr, rate, periods))


def convertible_bond(face, cr, rate, periods, conv_ratio, conv_price):
    return max(coupon_bond(face, cr, rate, periods), conv_ratio * conv_price)


def convertible_bond_lattice(face, coupon_rate, rate, periods, conv_ratio, stock0, vol,
                             dividend=0.0, call_schedule=None, put_schedule=None):
    n = max(1, int(periods))
    call_schedule = call_schedule or {}
    put_schedule = put_schedule or {}
    coupon = face * coupon_rate
    dt = 1.0
    if vol == 0:
        growth = math.exp((rate - dividend) * dt)
        terminal_stock = stock0 * growth ** n
        cont = [max(face, conv_ratio * terminal_stock)] * (n + 1)
        if n in call_schedule:
            cont = [min(v, call_schedule[n]) for v in cont]
        if n in put_schedule:
            cont = [max(v, put_schedule[n]) for v in cont]
        for step in range(n - 1, -1, -1):
            cont = [(coupon + cont[i]) / max(1.0 + rate, EPS) for i in range(len(cont) - 1)]
            ex = step + 1
            if ex in call_schedule:
                cont = [min(v, call_schedule[ex]) for v in cont]
            if ex in put_schedule:
                cont = [max(v, put_schedule[ex]) for v in cont]
        return cont[0]

    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend) * dt)
    p = min(max((growth - down) / max(up - down, EPS), 0.0), 1.0)

    def stock_nodes(step):
        return [stock0 * up ** (step - j) * down ** j for j in range(step + 1)]

    cont = [max(face, conv_ratio * s) for s in stock_nodes(n)]
    if n in call_schedule:
        cont = [min(v, call_schedule[n]) for v in cont]
    if n in put_schedule:
        cont = [max(v, put_schedule[n]) for v in cont]
    for step in range(n - 1, -1, -1):
        nodes = stock_nodes(step)
        expected_next = [p * cont[i] + (1 - p) * cont[i + 1] for i in range(len(cont) - 1)]
        hold = [(coupon + expected_next[i]) / max(1.0 + rate, EPS) for i in range(len(expected_next))]
        cont = [max(hold[i], conv_ratio * nodes[i]) for i in range(step + 1)]
        ex = step + 1
        if ex in call_schedule:
            cont = [min(v, call_schedule[ex]) for v in cont]
        if ex in put_schedule:
            cont = [max(v, put_schedule[ex]) for v in cont]
    return cont[0]


def stochastic_rate_bond(face, coupon_rate, rates):
    n = len(rates)
    coupon = face * coupon_rate
    dfs = []
    d = 1.0
    for i in range(n):
        d *= 1 + rates[i]
        dfs.append(d)
    s = sum(coupon / d for d in dfs)
    return s + face / dfs[-1]


if __name__ == "__main__":
    print("zero_coupon  ", repr(zero_coupon_bond(1000.0, 0.05, 3.0)))
    print("coupon_bond  ", repr(coupon_bond(1000.0, 0.05, 0.04, 5)))
    print("callable     ", repr(callable_bond_lattice(1000.0, 0.06, 0.05, 5, 1050.0, 3, 0.1)))
    print("putable      ", repr(putable_bond_lattice(1000.0, 0.06, 0.05, 5, 950.0, 3, 0.1)))
    print("conv_simple  ", repr(convertible_bond(100.0, 0.05, 0.05, 4, 2.0, 60.0)))
    print("conv_lat_vol ", repr(convertible_bond_lattice(
        100.0, 0.05, 0.05, 4, 2.0, 60.0, 0.2, dividend=0.01,
        call_schedule={3: 110.0}, put_schedule={2: 90.0})))
    print("conv_lat_vol0", repr(convertible_bond_lattice(
        100.0, 0.05, 0.05, 4, 2.0, 60.0, 0.0, dividend=0.01,
        call_schedule={3: 110.0}, put_schedule={2: 90.0})))
    print("stoch_rate   ", repr(stochastic_rate_bond(1000.0, 0.04, [0.03, 0.035, 0.04, 0.045])))
