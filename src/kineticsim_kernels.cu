#include "common.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace ks;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(_e) + " @ " +          \
                                     __FILE__ + ":" + std::to_string(__LINE__));\
        }                                                                       \
    } while (0)

__global__ void simulate_kernel(SimParams P,
                                const int* __restrict__ atype,
                                float*  __restrict__ d_bid_out,
                                float*  __restrict__ d_ask_out,
                                float*  __restrict__ d_lastprice,
                                double* __restrict__ d_volume,
                                int*    __restrict__ d_trades,
                                float*  __restrict__ d_pricehist) {
    const int market = blockIdx.x;
    const int t      = threadIdx.x;
    const int L      = P.n_levels;
    const int A      = P.n_agents;


    extern __shared__ float smem[];
    float* s_bid   = smem;
    float* s_ask   = s_bid   + L;
    float* s_BUY   = s_ask   + L;
    float* s_SELL  = s_BUY   + L;
    float* s_Dcum  = s_SELL  + L;
    float* s_Scum  = s_Dcum  + L;
    float* s_match = s_Scum  + L;
    int*   s_idx   = (int*)(s_match + L);

    __shared__ int   s_bestbid, s_bestask, s_pstar;
    __shared__ float s_mid, s_ret, s_prevmid, s_lastprice, s_V;
    __shared__ double s_vol;
    __shared__ int   s_trades;


    s_bid[t] = 0.0f;
    s_ask[t] = 0.0f;
    if (t == 0) {
        s_prevmid   = (float)P.init_price;
        s_lastprice = (float)P.init_price;
        s_vol       = 0.0;
        s_trades    = 0;
    }
    __syncthreads();
    if (t == P.init_price - 1) s_bid[t] = P.init_depth;
    if (t == P.init_price + 1) s_ask[t] = P.init_depth;
    __syncthreads();


    for (int step = 0; step < P.n_steps; ++step) {

        if (t == 0) { s_bestbid = -1; s_bestask = L; }
        __syncthreads();
        if (s_bid[t] > 0.0f) atomicMax(&s_bestbid, t);
        if (s_ask[t] > 0.0f) atomicMin(&s_bestask, t);
        __syncthreads();

        if (t == 0) {
            float mid = (s_bestbid >= 0 && s_bestask < L)
                            ? 0.5f * (float)(s_bestbid + s_bestask)
                            : s_lastprice;
            s_mid = mid;
            float r = 0.0f;
            if (mid > s_prevmid) r = 1.0f;
            else if (mid < s_prevmid) r = -1.0f;
            s_ret     = r;
            s_prevmid = mid;
        }
        __syncthreads();
        const float mid = s_mid;
        const float ret = s_ret;


        s_BUY[t]  = s_bid[t];
        s_SELL[t] = s_ask[t];
        __syncthreads();


        for (int a = t; a < A; a += L) {
            const int aty = atype[market * A + a];
            const Order o = decide(P, market, a, step, mid, ret, aty);
            if (o.side > 0) atomicAdd(&s_BUY[o.price],  o.qty);
            else            atomicAdd(&s_SELL[o.price], o.qty);
        }
        __syncthreads();


        s_Scum[t] = s_SELL[t];
        s_Dcum[t] = s_BUY[t];
        __syncthreads();
        for (int off = 1; off < L; off <<= 1) {
            float addS = (t >= off)     ? s_Scum[t - off] : 0.0f;
            float addD = (t + off < L)  ? s_Dcum[t + off] : 0.0f;
            __syncthreads();
            s_Scum[t] += addS;
            s_Dcum[t] += addD;
            __syncthreads();
        }


        s_match[t] = fminf(s_Dcum[t], s_Scum[t]);
        s_idx[t]   = t;
        __syncthreads();
        for (int stride = L >> 1; stride > 0; stride >>= 1) {
            if (t < stride) {
                float a0 = s_match[t], b0 = s_match[t + stride];
                if (b0 > a0 || (b0 == a0 && s_idx[t + stride] < s_idx[t])) {
                    s_match[t] = b0;
                    s_idx[t]   = s_idx[t + stride];
                }
            }
            __syncthreads();
        }
        if (t == 0) { s_V = s_match[0]; s_pstar = s_idx[0]; }
        __syncthreads();
        const float V = s_V;


        const float tb = fminf(fmaxf(V - (s_Dcum[t] - s_BUY[t]),  0.0f), s_BUY[t]);
        const float ts = fminf(fmaxf(V - (s_Scum[t] - s_SELL[t]), 0.0f), s_SELL[t]);
        s_bid[t] = s_BUY[t]  - tb;
        s_ask[t] = s_SELL[t] - ts;

        if (t == 0) {
            if (V > 0.0f) { s_lastprice = (float)s_pstar; s_trades++; }
            s_vol += (double)V;



            if (d_pricehist) d_pricehist[(size_t)step * gridDim.x + market] = s_lastprice;
        }
        __syncthreads();
    }


    d_bid_out[market * L + t] = s_bid[t];
    d_ask_out[market * L + t] = s_ask[t];
    if (t == 0) {
        d_lastprice[market] = s_lastprice;
        d_volume[market]    = s_vol;
        d_trades[market]    = s_trades;
    }
}

static py::dict simulate(int n_markets, int n_agents, int n_levels, int n_steps,
                         uint64_t seed, int init_price, float init_depth,
                         float max_order_qty, int noise_spread,
                         float market_order_prob, int maker_half_spread,
                         py::array_t<int, py::array::c_style | py::array::forcecast> agent_type,
                         bool record_prices = false) {
    if (n_levels > MAX_LEVELS || (n_levels & (n_levels - 1)) != 0)
        throw std::runtime_error("n_levels must be a power of two <= 1024");
    if ((size_t)agent_type.size() != (size_t)n_markets * n_agents)
        throw std::runtime_error("agent_type must have n_markets*n_agents entries");

    SimParams P;
    P.n_markets = n_markets; P.n_agents = n_agents; P.n_levels = n_levels;
    P.n_steps = n_steps; P.seed = seed; P.init_price = init_price;
    P.init_depth = init_depth; P.max_order_qty = max_order_qty;
    P.noise_spread = noise_spread; P.market_order_prob = market_order_prob;
    P.maker_half_spread = maker_half_spread;

    const int M = n_markets, L = n_levels, A = n_agents;
    const size_t book_elems = (size_t)M * L;

    int    *d_atype = nullptr, *d_trades = nullptr;
    float  *d_bid = nullptr, *d_ask = nullptr, *d_lastprice = nullptr;
    float  *d_pricehist = nullptr;
    double *d_vol = nullptr;

    CUDA_CHECK(cudaMalloc(&d_atype,    (size_t)M * A * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bid,      book_elems   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ask,      book_elems   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lastprice, (size_t)M   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vol,      (size_t)M    * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trades,   (size_t)M    * sizeof(int)));
    if (record_prices)
        CUDA_CHECK(cudaMalloc(&d_pricehist, (size_t)n_steps * M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_atype, agent_type.data(),
                          (size_t)M * A * sizeof(int), cudaMemcpyHostToDevice));

    const size_t shmem = (size_t)L * (7 * sizeof(float) + sizeof(int));
    if (shmem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(simulate_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    simulate_kernel<<<M, L, shmem>>>(P, d_atype, d_bid, d_ask,
                                     d_lastprice, d_vol, d_trades, d_pricehist);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));


    auto last  = py::array_t<float>(M);
    auto vol   = py::array_t<double>(M);
    auto trd   = py::array_t<int>(M);
    CUDA_CHECK(cudaMemcpy(last.mutable_data(), d_lastprice, M * sizeof(float),  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vol.mutable_data(),  d_vol,       M * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trd.mutable_data(),  d_trades,    M * sizeof(int),    cudaMemcpyDeviceToHost));

    size_t mem_bytes = ((size_t)M * A * sizeof(int)) + 2 * book_elems * sizeof(float)
                       + (size_t)M * (sizeof(float) + sizeof(double) + sizeof(int));

    py::dict out;
    if (record_prices) {
        auto hist = py::array_t<float>({(py::ssize_t)n_steps, (py::ssize_t)M});
        CUDA_CHECK(cudaMemcpy(hist.mutable_data(), d_pricehist,
                              (size_t)n_steps * M * sizeof(float), cudaMemcpyDeviceToHost));
        out["price_history"] = hist;
        mem_bytes += (size_t)n_steps * M * sizeof(float);
        cudaFree(d_pricehist);
    }

    cudaFree(d_atype); cudaFree(d_bid); cudaFree(d_ask);
    cudaFree(d_lastprice); cudaFree(d_vol); cudaFree(d_trades);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    out["last_price"]   = last;
    out["total_volume"] = vol;
    out["n_trades"]     = trd;
    out["elapsed_s"]    = (double)ms / 1000.0;
    out["gpu_mem_gb"]   = (double)mem_bytes / 1e9;
    return out;
}

PYBIND11_MODULE(kineticsim_cuda, m) {
    m.doc() = "KineticSim optimized custom CUDA engine (shared-memory persistent "
              "call-auction LOB).";
    m.def("simulate", &simulate,
          py::arg("n_markets"), py::arg("n_agents"), py::arg("n_levels"),
          py::arg("n_steps"), py::arg("seed"), py::arg("init_price"),
          py::arg("init_depth"), py::arg("max_order_qty"), py::arg("noise_spread"),
          py::arg("market_order_prob"), py::arg("maker_half_spread"),
          py::arg("agent_type"), py::arg("record_prices") = false,
          "Run the full simulation on the GPU and return final per-market stats "
          "(optionally with the (n_steps, n_markets) last-price trace).");
}
