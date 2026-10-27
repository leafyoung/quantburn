# quantburn 4-backend benchmark results

Post-review run (after the bench pass-criterion fix and the P2/P3 cleanups).
Machine: AMD Strix Halo (Radeon 8060S iGPU, gfx1151), Linux. 10 timed iters, 2 warmups. Median wall ms.

## Timings

| workload | torch CPU | torch ROCm (HIP) | burn CPU (NdArray) | burn Vulkan (wgpu) |
|---|---|---|---|---|
| W1 bsm_grid | 0.99 | 173.80 | 1.32 | 0.45 |
| W2 implied_vol | 12.65 | 15.72 | 4.89 | 21.31 |
| W3 binomial_american | 113.09 | 134.66 | 9.46 | 144.36 |
| W4 mc_european | 50.25 | 87.06 | 5481.75 | 39.92 |
| W5 mc_asian | 8.61 | 115.48 | 1055.09 | 10.84 |
| W6 heston | 92.86 | 2302.79 | 12209.29 | 39.19 |
| W7 cir_sim | 76.71 | 11499.07 | 18636.96 | 52.60 |
| W8 greeks_fd | 228.77 | 375.73 | 15937.12 | 52.86 |
| W9 var_es | 517.90 | 441.86 | 46.66 | 45.92 |
| W10 mlp | 563.80 | 1800.92 | 4958.84 | 48.55 |

## Correctness (check mode)

Pass criterion: global relative diff vs the f32 CPU reference (per DESIGN/bench_quantorch.py);
the anchor column is the same-metric diff vs an f64 reference. **All four backends: ALL PASS.**

| workload | torch CPU pass/global-rel | torch ROCm (HIP) pass/global-rel | burn CPU (NdArray) pass/global-rel | burn Vulkan (wgpu) pass/global-rel |
|---|---|---|---|---|
| W1 bsm_grid | PASS/0.0e+00 | PASS/6.8e-07 | PASS/0.0e+00 | PASS/5.8e-07 |
| W2 implied_vol | PASS/0.0e+00 | PASS/5.8e-07 | PASS/0.0e+00 | PASS/1.3e-06 |
| W3 binomial_american | PASS/0.0e+00 | PASS/6.6e-05 | PASS/0.0e+00 | PASS/0.0e+00 |
| W4 mc_european | PASS/0.0e+00 | PASS/9.1e-08 | PASS/0.0e+00 | PASS/5.8e-07 |
| W5 mc_asian | PASS/0.0e+00 | PASS/8.3e-08 | PASS/0.0e+00 | PASS/4.2e-07 |
| W6 heston | PASS/0.0e+00 | PASS/9.0e-08 | PASS/0.0e+00 | PASS/4.9e-07 |
| W7 cir_sim | PASS/0.0e+00 | PASS/7.4e-08 | PASS/0.0e+00 | PASS/9.7e-09 |
| W8 greeks_fd | PASS/0.0e+00 | PASS/3.7e-07 | PASS/0.0e+00 | PASS/8.8e-06 |
| W9 var_es | PASS/0.0e+00 | PASS/8.9e-08 | PASS/0.0e+00 | PASS/0.0e+00 |
| W10 mlp | PASS/0.0e+00 | PASS/1.9e-09 | PASS/0.0e+00 | PASS/7.8e-08 |

Notes:
- torch-cpu and burn-cpu are the f32 reference itself, hence global-rel 0.
- burn-cpu medians on the tensor-heavy workloads (W4/W6/W7/W8/W10) are noisier
  and slower than the pre-review run — the NdArray path is single-threaded and
  this run shared the machine; treat burn-CPU numbers as order-of-magnitude.
- Known performance shape: burn-Vulkan fastest on tensor-heavy workloads;
  torch-ROCm kernel-launch-latency-bound on 250-step sequential sims;
  W10's Python-vs-Rust absolute values differ by design (documented in AGENTS.md).
