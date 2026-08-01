# KineticSim

**A lightweight, high-performance execution engine for real-time market simulators.**

KineticSim simulates a large *ensemble* of agent-based financial markets — many
independent limit-order books, each populated by many heterogeneous agents — on
a single GPU using hand-written CUDA kernels. It targets the high-frequency /
large-scale agent-based simulation setting (cf. *Agent-Based Financial Systems*
and *Agent-Based Simulation for Market Design*), where throughput is the
bottleneck and GPU acceleration is the standard remedy (e.g. JaxMARL-HFT).

This repository contains the optimized engine **and three comparison backends**
implementing the *identical* market model, so the only thing that varies across
them is performance:

| Backend | Role | Where |
|---|---|---|
| **KineticSim (optimized CUDA)** | **this work** | `src/kineticsim_kernels.cu` |
| Naive custom CUDA | baseline 1 — "a CUDA kernel, but unoptimized" | `src/naive_kernels.cu` |
| PyTorch GPU | baseline 2 — "GPU, but no custom kernels" | `python/kineticsim/torch_gpu.py` |
| CPU (NumPy) | reference oracle / extra baseline | `python/kineticsim/reference_cpu.py` |

Target hardware: **NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB)** + **AMD Ryzen 9
9950X3D**.

---

## The simulation model (shared by every backend)

Discrete-time, price-grid, **uniform-price call auction** over a persistent
aggregate limit-order book. Each of `M` markets keeps resting buy/sell quantity
on a grid of `L` price ticks; each step its `A` agents observe the book, submit
one order, and the book is cleared:

```
Dcum[p] = Σ_{q ≥ p} BUY[q]        (demand willing at price p)
Scum[p] = Σ_{q ≤ p} SELL[q]       (supply willing at price p)
p*      = argmax_p min(Dcum[p], Scum[p])      (volume-maximizing clearing price)
V       = min(Dcum[p*], Scum[p*])             (executed volume)
```

Highest-limit buyers and lowest-limit sellers fill first; unmatched interest
persists as the next step's resting book, so spread, depth and price dynamics
emerge endogenously. Agents come in three strategy classes — **noise**,
**momentum**, and **market-maker** (liquidity provider).

This formulation is deliberately chosen because it is (a) a legitimate
market-design mechanism (frequent batch auctions, *Budish–Cramton–Shim*), and
(b) expressible purely as prefix-sums over the price grid — so it can be
implemented identically on CPU, in vectorized PyTorch, and in hand-written CUDA.
That identity is what makes the cross-backend speedup comparison fair. The model
is the executable specification in `python/kineticsim/model.py` and
`python/kineticsim/reference_cpu.py`.

---

## What makes the optimized engine fast

The optimized kernel (`src/kineticsim_kernels.cu`) applies the following, none of
which the baselines use:

1. **One block per market, one thread per price level** — natural mapping that
   makes the auction scans and reductions intra-block and synchronization-free
   across markets.
2. **Shared-memory-resident book, persistent across the whole run** — a *single*
   kernel launch executes all `n_steps`; the order book never round-trips to
   global memory between steps, and launch overhead is paid once instead of
   `n_steps` times.
3. **Shared-memory atomic order aggregation** for scattering agent orders onto
   the grid.
4. **Parallel Hillis–Steele prefix/suffix scans + a block argmax reduction** to
   clear the auction (no serial `O(L)` passes).
5. **Stateless counter-based RNG** (SplitMix64) — removes per-agent cuRAND state
   storage and initialization cost.
6. **Structure-of-Arrays, coalesced** agent metadata access.

The naive CUDA baseline deliberately does the opposite of every point above
(one thread per market, global-memory book, per-step launches, serial scans) so
the benchmark isolates exactly what the kernel engineering buys you. Because both
CUDA backends share the same decision function and RNG (`src/common.cuh`), they
produce **bitwise-identical** results — the difference is pure speed.

---

## Build

Requirements: CUDA Toolkit **≥ 12.8** (needed for Blackwell sm_120), CMake ≥ 3.24,
a C++17 host compiler, Python ≥ 3.9, and `pip install pybind11 numpy`.

```bash
cd code
pip install -r requirements.txt
./build.sh                 # builds for sm_120 by default
# other GPUs:  KS_ARCH=89 ./build.sh   (Ada / RTX 40xx)
```

This compiles `kineticsim_cuda` and `kineticsim_naive` and drops the `.so` files
into `python/`, next to the `kineticsim` package, so they import directly.

For the PyTorch baseline, install a CUDA build of torch matching your driver:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The CPU reference needs only NumPy and runs without any of the above.

---

## Run

```bash
# correctness first: clearing-engine fuzz + CUDA==naive (exact) + CUDA≈CPU (stats)
python3 benchmarks/validate_correctness.py

# full evaluation across all available backends (auto-detected)
python3 benchmarks/run_benchmarks.py --all --trials 7

# figures from the resulting CSV
python3 benchmarks/plot_results.py
```

`run_benchmarks.py` writes `benchmarks/results.csv` (one row per measured run,
with speedup columns vs CPU and vs PyTorch) and prints a summary table.
`plot_results.py` writes PNG+PDF figures to `benchmarks/figures/`.

Useful flags: `--markets` / `--agents` / `--fixed` / `--latency` to run one
sweep; `--backends cpu,naive,cuda`; `--steps`, `--markets-n`, `--agents-n`,
`--trials`; `--cpu-max-events` to cap the (slow) CPU backend on huge workloads.

### Evaluation produced for the paper

* **markets sweep** — strong scaling in `M ∈ {64…16384}` (throughput vs parallel
  markets).
* **agents sweep** — scaling in `A ∈ {16…1024}` (compute density per market).
* **fixed workload** — head-line wall-clock and speedup factors on one large run.
* **latency** — per-step latency (median + min/max over repeated trials).

Reported metrics per run: agent-events/s, steps/s, µs/step, ns/event, wall-clock
(median/min/max/std over trials), GPU memory, speedup vs CPU and vs PyTorch, plus
aggregate market statistics (mean/std clearing price, volume, trades) used to
confirm the backends agree.

---

## Layout

```
code/
├── src/
│   ├── common.cuh             shared params, RNG, agent decision (one source of truth)
│   ├── kineticsim_kernels.cu  OPTIMIZED engine + pybind11 module  (this work)
│   └── naive_kernels.cu       naive CUDA baseline + pybind11 module
├── python/kineticsim/
│   ├── model.py               SimConfig + model documentation
│   ├── reference_cpu.py       vectorized NumPy reference (oracle)
│   ├── torch_gpu.py           PyTorch GPU baseline
│   └── cuda_backend.py        wrappers around the compiled CUDA modules
├── benchmarks/
│   ├── run_benchmarks.py      sweeps -> results.csv
│   ├── plot_results.py        results.csv -> figures
│   └── validate_correctness.py
├── tests/test_model.py
├── CMakeLists.txt   build.sh   requirements.txt
```

## Reproducibility & validation notes

* The clearing engine is fuzz-tested for volume conservation and book
  non-negativity (`tests/`, `validate_correctness.py`).
* Optimized-vs-naive CUDA agreement is **exact** (shared RNG), so any future
  kernel change that alters results is caught immediately.
* CUDA-vs-CPU agreement is **statistical** (different RNG streams across the
  Python/CUDA boundary): aggregate market statistics are required to match
  within 10%.
* All GPU timing uses CUDA events with an explicit device synchronize; GPU
  backends are warmed up before measurement and every config is repeated
  `--trials` times with the median reported.
  
## Citation
```bash
@article{jayakody2026kineticsim,
  title={KineticSim: A Lightweight, High-Performance Execution Engine for Real-Time Market Simulators},
  author={Jayakody, Shakya and Jayakody, Prarthinie},
  journal={arXiv preprint arXiv:2606.21784},
  year={2026}
}
```
