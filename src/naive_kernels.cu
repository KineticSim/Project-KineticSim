#include "common.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace ks;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                cudaGetErrorString(_e) + " @ " + __FILE__ + ":" +              \
                std::to_string(__LINE__));                                      \
    } while (0)

__global__ void init_kernel(SimParams P, float* bid, float* ask,
                            float* lastprice, float* prevmid,
                            double* vol, int* trades) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= P.n_markets) return;
    const int L = P.n_levels;
    for (int p = 0; p < L; ++p) { bid[m * L + p] = 0.0f; ask[m * L + p] = 0.0f; }
    bid[m * L + P.init_price - 1] = P.init_depth;
    ask[m * L + P.init_price + 1] = P.init_depth;
    lastprice[m] = (float)P.init_price;
    prevmid[m]   = (float)P.init_price;
    vol[m]       = 0.0;
    trades[m]    = 0;
}

__global__ void step_kernel(SimParams P, int step, const int* __restrict__ atype,
                            float* bid, float* ask, float* BUY, float* SELL,
                            float* scanB, float* scanS,
                            float* lastprice, float* prevmid,
                            double* vol, int* trades) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= P.n_markets) return;
    const int L = P.n_levels, A = P.n_agents;
    float* bid_m   = bid   + (size_t)m * L;
    float* ask_m   = ask   + (size_t)m * L;
    float* BUY_m   = BUY   + (size_t)m * L;
    float* SELL_m  = SELL  + (size_t)m * L;
    float* scanB_m = scanB + (size_t)m * L;
    float* scanS_m = scanS + (size_t)m * L;


    int bb = -1, ba = L;
    for (int p = 0; p < L; ++p) if (bid_m[p] > 0.0f) bb = p;
    for (int p = 0; p < L; ++p) if (ask_m[p] > 0.0f) { ba = p; break; }

    float lastp = lastprice[m];
    float mid = (bb >= 0 && ba < L) ? 0.5f * (float)(bb + ba) : lastp;
    float pm  = prevmid[m];
    float ret = (mid > pm) ? 1.0f : (mid < pm ? -1.0f : 0.0f);
    prevmid[m] = mid;


    for (int p = 0; p < L; ++p) { BUY_m[p] = bid_m[p]; SELL_m[p] = ask_m[p]; }


    for (int a = 0; a < A; ++a) {
        const int aty = atype[(size_t)m * A + a];
        const Order o = decide(P, m, a, step, mid, ret, aty);
        if (o.side > 0) BUY_m[o.price]  += o.qty;
        else            SELL_m[o.price] += o.qty;
    }


    float run = 0.0f;
    for (int p = 0; p < L; ++p)       { run += SELL_m[p]; scanS_m[p] = run; }
    run = 0.0f;
    for (int p = L - 1; p >= 0; --p)  { run += BUY_m[p];  scanB_m[p] = run; }


    float V = 0.0f; int ps = 0;
    for (int p = 0; p < L; ++p) {
        float mm = fminf(scanB_m[p], scanS_m[p]);
        if (mm > V) { V = mm; ps = p; }
    }


    for (int p = 0; p < L; ++p) {
        float tb = fminf(fmaxf(V - (scanB_m[p] - BUY_m[p]), 0.0f), BUY_m[p]);
        float ts = fminf(fmaxf(V - (scanS_m[p] - SELL_m[p]), 0.0f), SELL_m[p]);
        bid_m[p] = BUY_m[p]  - tb;
        ask_m[p] = SELL_m[p] - ts;
    }

    if (V > 0.0f) { lastprice[m] = (float)ps; trades[m] += 1; }
    vol[m] += (double)V;
}

static py::dict simulate(int n_markets, int n_agents, int n_levels, int n_steps,
                         uint64_t seed, int init_price, float init_depth,
                         float max_order_qty, int noise_spread,
                         float market_order_prob, int maker_half_spread,
                         py::array_t<int, py::array::c_style | py::array::forcecast> agent_type) {
    if (n_levels > MAX_LEVELS)
        throw std::runtime_error("n_levels too large");
    if ((size_t)agent_type.size() != (size_t)n_markets * n_agents)
        throw std::runtime_error("agent_type must have n_markets*n_agents entries");

    SimParams P;
    P.n_markets = n_markets; P.n_agents = n_agents; P.n_levels = n_levels;
    P.n_steps = n_steps; P.seed = seed; P.init_price = init_price;
    P.init_depth = init_depth; P.max_order_qty = max_order_qty;
    P.noise_spread = noise_spread; P.market_order_prob = market_order_prob;
    P.maker_half_spread = maker_half_spread;

    const int M = n_markets, L = n_levels, A = n_agents;
    const size_t be = (size_t)M * L;

    int    *d_atype, *d_trades;
    float  *d_bid, *d_ask, *d_BUY, *d_SELL, *d_scanB, *d_scanS, *d_last, *d_prev;
    double *d_vol;
    CUDA_CHECK(cudaMalloc(&d_atype, (size_t)M * A * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bid,   be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ask,   be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_BUY,   be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_SELL,  be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scanB, be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scanS, be * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last,  (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prev,  (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vol,   (size_t)M * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trades,(size_t)M * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_atype, agent_type.data(),
                          (size_t)M * A * sizeof(int), cudaMemcpyHostToDevice));

    const int block = 128;
    const int grid  = (M + block - 1) / block;

    init_kernel<<<grid, block>>>(P, d_bid, d_ask, d_last, d_prev, d_vol, d_trades);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int step = 0; step < n_steps; ++step) {
        step_kernel<<<grid, block>>>(P, step, d_atype, d_bid, d_ask, d_BUY,
                                     d_SELL, d_scanB, d_scanS, d_last, d_prev,
                                     d_vol, d_trades);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    auto last = py::array_t<float>(M);
    auto vol  = py::array_t<double>(M);
    auto trd  = py::array_t<int>(M);
    CUDA_CHECK(cudaMemcpy(last.mutable_data(), d_last, M * sizeof(float),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vol.mutable_data(),  d_vol,  M * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trd.mutable_data(),  d_trades, M * sizeof(int),  cudaMemcpyDeviceToHost));

    size_t mem_bytes = (size_t)M * A * sizeof(int) + 6 * be * sizeof(float)
                       + (size_t)M * (2 * sizeof(float) + sizeof(double) + sizeof(int));

    cudaFree(d_atype); cudaFree(d_bid); cudaFree(d_ask); cudaFree(d_BUY);
    cudaFree(d_SELL); cudaFree(d_scanB); cudaFree(d_scanS); cudaFree(d_last);
    cudaFree(d_prev); cudaFree(d_vol); cudaFree(d_trades);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    py::dict out;
    out["last_price"]   = last;
    out["total_volume"] = vol;
    out["n_trades"]     = trd;
    out["elapsed_s"]    = (double)ms / 1000.0;
    out["gpu_mem_gb"]   = (double)mem_bytes / 1e9;
    return out;
}

PYBIND11_MODULE(kineticsim_naive, m) {
    m.doc() = "KineticSim naive custom CUDA baseline (one thread per market, "
              "global memory, per-step launches).";
    m.def("simulate", &simulate,
          py::arg("n_markets"), py::arg("n_agents"), py::arg("n_levels"),
          py::arg("n_steps"), py::arg("seed"), py::arg("init_price"),
          py::arg("init_depth"), py::arg("max_order_qty"), py::arg("noise_spread"),
          py::arg("market_order_prob"), py::arg("maker_half_spread"),
          py::arg("agent_type"));
}
