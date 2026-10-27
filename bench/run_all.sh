#!/usr/bin/env bash
# Run the 4-backend benchmark + check suite (see bench/DESIGN.md).
#   torch CPU  | torch ROCm | burn CPU | burn Vulkan
# Writes JSON per backend into bench/results/ and a summary bench/results.md.
set -uo pipefail
cd "$(dirname "$0")/.."

ITERS="${ITERS:-10}"
OUT=bench/results
mkdir -p "$OUT" bench/data

# 1. shared inputs (idempotent)
[ -f bench/data/shapes.json ] || python3 bench/gen_inputs.py

# 2. python torch CPU
echo "=== torch-cpu ==="
python3 bench/bench_quantorch.py --mode check --device cpu --json "$OUT/check_torch_cpu.json"
python3 bench/bench_quantorch.py --mode bench --device cpu --iters "$ITERS" --json "$OUT/bench_torch_cpu.json"

# 3. python torch ROCm (rl-0dte venv: torch+rocm7.13, gfx1151)
ROCM_PY="$HOME/devv/study/study-rl-0dte/.venv/bin/python"
if [ -x "$ROCM_PY" ]; then
  echo "=== torch-rocm ==="
  "$ROCM_PY" bench/bench_quantorch.py --mode check --device cuda --json "$OUT/check_torch_rocm.json"
  "$ROCM_PY" bench/bench_quantorch.py --mode bench --device cuda --iters "$ITERS" --json "$OUT/bench_torch_rocm.json"
else
  echo "=== torch-rocm: venv not found, skipped ==="
fi

# 4. burn CPU
echo "=== burn-cpu ==="
cargo run --release --bin bench -- --mode check --backend cpu --json "$OUT/check_burn_cpu.json"
cargo run --release --bin bench -- --mode bench --backend cpu --iters "$ITERS" --json "$OUT/bench_burn_cpu.json"

# 5. burn Vulkan
echo "=== burn-vulkan ==="
cargo run --release --features vulkan --bin bench -- --mode check --backend vulkan --json "$OUT/check_burn_vulkan.json"
cargo run --release --features vulkan --bin bench -- --mode bench --backend vulkan --iters "$ITERS" --json "$OUT/bench_burn_vulkan.json"

echo "JSON results in $OUT/"
