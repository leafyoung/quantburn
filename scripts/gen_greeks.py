"""Reference generator for quantburn/src/malliavin.rs tests.

Re-implements torchquantlib/core/risk/greeks/malliavin.py in pure float64,
drawing normals from scripts/refgen.py with the SAME seed as the Rust code
(3001). Prints expected values to embed in the Rust tests.
"""
import math
import sys

sys.path.insert(0, "scripts")
from refgen import normals

SEED = 3001


def mean(xs):
    return sum(xs) / len(xs)


def european(s0, k, t, r, sigma, n, seed):
    z = normals(seed, n)
    dt = t
    disc = math.exp(-r * t)
    d = {}
    sts = [s0 * math.exp((r - 0.5 * sigma**2) * t + sigma * math.sqrt(dt) * zi) for zi in z]
    pay = [max(s - k, 0.0) for s in sts]
    dts = [sts[i] * z[i] / math.sqrt(dt) for i in range(n)]
    d2s = [dts[i] * (z[i] / math.sqrt(dt) - sigma * math.sqrt(dt)) for i in range(n)]
    dsg = [sts[i] * (z[i] * math.sqrt(dt) - sigma * dt) for i in range(n)]
    drs = [s * t for s in sts]
    return {
        "Delta": disc * mean([p * w for p, w in zip(pay, dts)]) / (sigma * s0),
        "Gamma": disc * mean([p * w for p, w in zip(pay, d2s)]) / (sigma**2 * s0**2),
        "Vega": disc * mean([p * w for p, w in zip(pay, dsg)]) / sigma,
        "Theta": -disc * mean([p * ((r - 0.5 * sigma**2) * s + 0.5 * sigma * w)
                               for p, s, w in zip(pay, sts, dts)]) / t,
        "Rho": disc * mean([p * w for p, w in zip(pay, drs)]) - t * disc * mean(pay),
    }


def digital(s0, k, t, r, sigma, q, n, eps, seed):
    z = normals(seed, n)
    dt = t
    disc = math.exp(-r * t)
    norm = eps * math.sqrt(2.0 * math.pi)
    sts = [s0 * math.exp((r - 0.5 * sigma**2) * t + sigma * math.sqrt(dt) * zi) for zi in z]
    da = [(1.0 / norm) * math.exp(-0.5 * ((s - k) / eps) ** 2) for s in sts]
    dts = [sts[i] * z[i] / math.sqrt(dt) for i in range(n)]
    d2s = [dts[i] * (z[i] / math.sqrt(dt) - sigma * math.sqrt(dt)) for i in range(n)]
    dsg = [sts[i] * (z[i] * math.sqrt(dt) - sigma * dt) for i in range(n)]
    w = [q * d for d in da]
    return {
        "Delta": disc * mean([a * b for a, b in zip(w, dts)]) / (sigma * s0),
        "Gamma": disc * mean([a * b for a, b in zip(w, d2s)]) / (sigma**2 * s0**2),
        "Vega": disc * mean([a * b for a, b in zip(w, dsg)]) / sigma,
    }


def simulate_paths(s0, t, r, sigma, steps, z):
    dt = t / steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    out = []
    for row in z:
        path = [s0]
        for zi in row:
            path.append(path[-1] * math.exp(drift + vol * zi))
        out.append(path)
    return out


def draw_matrix(seed, n, steps):
    vals = normals(seed, n * steps)  # row-major like torch.randn((n, steps))
    return [vals[i * steps:(i + 1) * steps] for i in range(n)]


def barrier(s0, k, h, t, r, sigma, otype, n, steps, seed):
    z = draw_matrix(seed, n, steps)
    paths = simulate_paths(s0, t, r, sigma, steps, z)
    dt = t / steps
    disc = math.exp(-r * t)
    a = [0.0, 0.0, 0.0]
    for zi, sp in zip(z, paths):
        st = sp[-1]
        if otype == "up-and-out":
            hit = any(s >= h for s in sp[1:])
        else:
            hit = any(s <= h for s in sp[1:])
        if hit:
            continue
        pay = max(st - k, 0.0)
        zsum = sum(zi)
        dts = st * zsum / math.sqrt(dt)
        d2s = dts * (zsum / math.sqrt(dt) - sigma * math.sqrt(dt))
        dsg = st * (sum(v * v - 1.0 for v in zi) * math.sqrt(dt))
        a[0] += pay * dts
        a[1] += pay * d2s
        a[2] += pay * dsg
    m = n
    return {
        "Delta": disc * a[0] / m / (sigma * s0),
        "Gamma": disc * a[1] / m / (sigma**2 * s0**2),
        "Vega": disc * a[2] / m / sigma,
    }


def lookback(s0, t, r, sigma, n, steps, seed):
    z = draw_matrix(seed, n, steps)
    paths = simulate_paths(s0, t, r, sigma, steps, z)
    dt = t / steps
    disc = math.exp(-r * t)
    acc = 0.0
    for zi, sp in zip(z, paths):
        st = sp[-1]
        prefix = sp[:-1]
        smin = min(prefix)
        pay = max(st - smin, 0.0)
        dts = st * sum(zi) / math.sqrt(dt)
        num = 0.0
        cnt = 0
        for j, s in enumerate(prefix):
            if s == smin:
                num += s * zi[j] / math.sqrt(dt)
                cnt += 1
        dsmin = num / cnt if cnt else 0.0
        if pay > 0:
            acc += (dts - dsmin) / (sigma * s0)
    return disc * acc / n


def cholesky(m):
    nn = len(m)
    l = [[0.0] * nn for _ in range(nn)]
    for i in range(nn):
        for j in range(i + 1):
            s = m[i][j]
            for kk in range(j):
                s -= l[i][kk] * l[j][kk]
            l[i][j] = math.sqrt(s) if i == j else s / l[j][j]
    return l


def basket(s0, k, t, r, sigma, rho, weights, n, seed):
    na = len(s0)
    l = cholesky(rho)
    zflat = normals(seed, n * na)
    dt = t
    disc = math.exp(-r * t)
    acc = [[0.0, 0.0, 0.0] for _ in range(na)]
    for p in range(n):
        z = zflat[p * na:(p + 1) * na]
        st = []
        for j in range(na):
            zc = sum(z[kk] * l[j][kk] for kk in range(na))
            st.append(s0[j] * math.exp((r - 0.5 * sigma[j] ** 2) * t + sigma[j] * math.sqrt(dt) * zc))
        bp = sum(weights[j] * st[j] for j in range(na))
        pay = max(bp - k, 0.0)
        for j in range(na):
            zc = sum(z[kk] * l[j][kk] for kk in range(na))
            dts = st[j] * zc / math.sqrt(dt)
            d2s = dts * (zc / math.sqrt(dt) - sigma[j] * math.sqrt(dt))
            dsg = st[j] * (zc * math.sqrt(dt) - sigma[j] * dt)
            w = pay * weights[j]
            acc[j][0] += w * dts
            acc[j][1] += w * d2s
            acc[j][2] += w * dsg
    d = [disc * acc[j][0] / n / (sigma[j] * s0[j]) for j in range(na)]
    g = [disc * acc[j][1] / n / (sigma[j] ** 2 * s0[j] ** 2) for j in range(na)]
    v = [disc * acc[j][2] / n / sigma[j] for j in range(na)]
    return d, g, v


def asian(s0, k, t, r, sigma, n, steps, seed):
    z = draw_matrix(seed, n, steps)
    paths = simulate_paths(s0, t, r, sigma, steps, z)
    dt = t / steps
    disc = math.exp(-r * t)
    a = [0.0, 0.0, 0.0]
    for zi, sp in zip(z, paths):
        smean = sum(sp[1:]) / steps
        pay = max(smean - k, 0.0)
        dsm = sum(sp[j + 1] * zi[j] / math.sqrt(dt) for j in range(steps)) / steps
        d2m = sum(sp[j + 1] * zi[j] / math.sqrt(dt) * (zi[j] / math.sqrt(dt) - sigma * math.sqrt(dt))
                  for j in range(steps)) / steps
        dsgm = sum(sp[j + 1] * (zi[j] * math.sqrt(dt) - sigma * dt) for j in range(steps)) / steps
        a[0] += pay * dsm
        a[1] += pay * d2m
        a[2] += pay * dsgm
    return {
        "Delta": disc * a[0] / n / (sigma * s0),
        "Gamma": disc * a[1] / n / (sigma**2 * s0**2),
        "Vega": disc * a[2] / n / sigma,
    }


N = 20_000
STEPS = 25

print("european:", european(100.0, 100.0, 1.0, 0.05, 0.2, N, SEED))
print("digital:", digital(100.0, 100.0, 1.0, 0.05, 0.2, 10.0, N, 2.0, SEED))
print("barrier_up:", barrier(100.0, 110.0, 130.0, 1.0, 0.05, 0.2, "up-and-out", N, STEPS, SEED))
print("barrier_down:", barrier(100.0, 90.0, 80.0, 1.0, 0.05, 0.2, "down-and-out", N, STEPS, SEED))
print("lookback:", lookback(100.0, 1.0, 0.05, 0.2, N, STEPS, SEED))
print("basket:", basket([100.0, 90.0, 80.0], 95.0, 1.0, 0.05, [0.2, 0.25, 0.3],
                        [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]],
                        [0.4, 0.3, 0.3], N, SEED))
print("asian:", asian(100.0, 100.0, 1.0, 0.05, 0.2, N, STEPS, SEED))
