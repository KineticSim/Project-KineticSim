#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace ks {

constexpr int NOISE       = 0;
constexpr int MOMENTUM    = 1;
constexpr int MAKER       = 2;
constexpr int FUNDAMENTAL = 3;

constexpr int MAX_LEVELS = 1024;

struct SimParams {
    int   n_markets;
    int   n_agents;
    int   n_levels;
    int   n_steps;
    uint64_t seed;

    int   init_price;
    float init_depth;

    float max_order_qty;
    int   noise_spread;
    float market_order_prob;
    int   maker_half_spread;
};

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float urand(uint64_t seed, uint32_t gid,
                                        uint32_t step, uint32_t ch) {
    uint64_t k = seed ^ (0x123456789ABCDEFULL * (uint64_t)ch);
    k = splitmix64(k + 0x9E3779B97F4A7C15ULL * (uint64_t)gid);
    k = splitmix64(k + 0xD1B54A32D192ED03ULL * (uint64_t)step);

    return (float)((double)(k >> 11) * (1.0 / 9007199254740992.0));
}

struct Order {
    int   side;
    int   price;
    float qty;
};

__device__ __forceinline__ Order decide(const SimParams& P, int market, int a,
                                         int step, float mid, float ret,
                                         int atype) {
    const int   L = P.n_levels;
    const uint32_t gid = (uint32_t)market * (uint32_t)P.n_agents + (uint32_t)a;

    const float u_side = urand(P.seed, gid, step, 0);
    const float u_mkt  = urand(P.seed, gid, step, 1);
    const int   qty    = 1 + (int)(urand(P.seed, gid, step, 2) * P.max_order_qty);
    const int   noff   = -P.noise_spread +
                         (int)(urand(P.seed, gid, step, 3) * (2 * P.noise_spread + 1));

    int side, price;
    if (atype == MAKER) {
        const bool mmbuy = (((a + step) & 1) == 0);
        side  = mmbuy ? 1 : -1;
        price = (int)rintf(mid + (mmbuy ? -(float)P.maker_half_spread
                                        :  (float)P.maker_half_spread));
    } else {
        if (atype == MOMENTUM) {
            const int ns = (u_side < 0.5f) ? 1 : -1;
            side  = (ret != 0.0f) ? (int)ret : ns;
            price = (int)rintf(mid + (float)side);
        } else if (atype == FUNDAMENTAL) {


            const float dev = (float)P.init_price - mid;
            const int ns = (u_side < 0.5f) ? 1 : -1;
            side  = (dev > 0.0f) ? 1 : ((dev < 0.0f) ? -1 : ns);
            price = (int)rintf(mid + (float)side);
        } else {
            side  = (u_side < 0.5f) ? 1 : -1;
            price = (int)rintf(mid + (float)noff);
        }
        if (u_mkt < P.market_order_prob) {
            price = (side > 0) ? (L - 1) : 0;
        }
    }
    if (price < 0)      price = 0;
    if (price > L - 1)  price = L - 1;

    Order o; o.side = side; o.price = price; o.qty = (float)qty;
    return o;
}

}
