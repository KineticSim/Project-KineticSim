#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${KS_ARCH:-120}"
PYBIND_DIR="$(python3 -m pybind11 --cmakedir)"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
  -Dpybind11_DIR="${PYBIND_DIR}"

cmake --build build -j

echo
echo "Built modules into ./python :"
ls -1 python/*.so 2>/dev/null || echo "  (no .so found -- check the build output above)"
echo "Run the benchmarks with:  python3 benchmarks/run_benchmarks.py --all"
