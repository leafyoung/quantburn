#!/usr/bin/env python3
"""Generate shared benchmark inputs under bench/data/ per DESIGN.md.

Seed 7, torch CPU f64 normals cast to f32, C-order raw little-endian .f32
files plus shapes.json. Run once; all four backends read the same files.
"""
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SEED = 7

# name -> shape
SPECS = {
    "mc_euro_normals": [100_000, 100],
    "mc_asian_normals": [50_000, 64],
    "heston_zs": [20_000, 250],
    "heston_zv": [20_000, 250],
    "cir_normals": [50_000, 250],
    "var_losses": [1_000_000],
}

MLP_DIMS = [32] + [512] * 7 + [1]  # 8x512 MLP: 32->512->...->512->1


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    shapes = {}

    for name, shape in SPECS.items():
        t = torch.randn(shape, generator=g, dtype=torch.float64, device="cpu")
        t32 = t.to(torch.float32)
        path = os.path.join(DATA, f"{name}.f32")
        # C-order raw little-endian f32
        t32.contiguous().numpy().tofile(path)
        shapes[name] = {"shape": shape, "file": f"{name}.f32"}
        print(f"{name}: {shape} -> {os.path.getsize(path)} bytes (expect {4 * _numel(shape)})")

    # MLP weights: flat, layer-major [in_dim*512, 512*512*7, 512*1]
    n_weights = sum(a * b for a, b in zip(MLP_DIMS[:-1], MLP_DIMS[1:]))  # 32*512 + 512*512*7 + 512*1
    w = torch.randn([n_weights], generator=g, dtype=torch.float64, device="cpu").to(torch.float32)
    path = os.path.join(DATA, "mlp_weights.f32")
    w.contiguous().numpy().tofile(path)
    shapes["mlp_weights"] = {"shape": [n_weights], "file": "mlp_weights.f32",
                             "layer_dims": MLP_DIMS, "layout": "layer-major flat "
                             "[in_dim*512, 512*512*7, 512*1]"}
    print(f"mlp_weights: {list(w.shape)} ({MLP_DIMS}) -> {os.path.getsize(path)} bytes "
          f"(expect {4 * w.numel()})")

    with open(os.path.join(DATA, "shapes.json"), "w") as f:
        json.dump({"seed": SEED, "dtype": "float32", "order": "C", "endianness": "little",
                   "tensors": shapes}, f, indent=2)
    print("wrote shapes.json")


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


if __name__ == "__main__":
    main()
