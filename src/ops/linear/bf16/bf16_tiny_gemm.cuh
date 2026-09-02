#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct alignas(16) Bf16TinyPack8 {
    std::uint32_t words[4];
};
static_assert(sizeof(Bf16TinyPack8) == 16);

// Generic SIMT GEMM for small-n BF16 rows that fit no tuned GEMV/MMA tile family (currently the
// qwen4exp HC inject row [4, 10240]).
//
// out = weight * x; weight is [N,k] row-major. x is [t,k] token-major (token stride k) and out
// is [t,N] token-major (token stride N), matching every other production linear op. One block per
// token (blockIdx.x = tok); each thread accumulates all N rows over an 8-element K slice (weight
// loads vectorized 16B, activation loads scalar). kThreads * 8 must divide k evenly for the
// vectorized path; the scalar tail covers the remainder.
template <int NRows, int kThreads>
__global__ void bf16_tiny_gemm_kernel(const __nv_bfloat16* __restrict__ weight,
                                      const __nv_bfloat16* __restrict__ x,
                                      __nv_bfloat16* __restrict__ out, std::int32_t k) {
    static_assert(NRows > 0 && NRows <= 8, "tiny-n GEMM supports at most eight rows");
    static_assert(kThreads >= kWarpSize && (kThreads % kWarpSize) == 0);
    constexpr int kPack     = 8;
    constexpr int kKPerStep = kThreads * kPack;

    const std::int32_t tok  = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t lane = static_cast<int>(threadIdx.x);
    __shared__ float partials[kThreads / kWarpSize][NRows];

    // Token-major activation column base: element (tok, col) is at tok*k + col.
    const __nv_bfloat16* xtok = x + static_cast<std::int64_t>(tok) * k;
    const std::int64_t base   = static_cast<std::int64_t>(lane) * kPack;
    float acc[NRows]          = {};

    const std::int32_t full_k = (k / kKPerStep) * kKPerStep;
    for (std::int32_t i0 = 0; i0 < full_k; i0 += kKPerStep) {
        const std::int64_t k0 = i0 + base;
#pragma unroll
        for (int r = 0; r < NRows; ++r) {
            const Bf16TinyPack8 wp =
                load_vec<Bf16TinyPack8>(weight + static_cast<std::int64_t>(r) * k + k0);
            const float2 w0 = bf16x2_bits_to_float2(wp.words[0]);
            const float2 w1 = bf16x2_bits_to_float2(wp.words[1]);
            const float2 w2 = bf16x2_bits_to_float2(wp.words[2]);
            const float2 w3 = bf16x2_bits_to_float2(wp.words[3]);
            const float w[8] = {w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y};
#pragma unroll
            for (int j = 0; j < kPack; ++j) {
                acc[r] = fmaf(w[j], __bfloat162float(xtok[k0 + j]), acc[r]);
            }
        }
    }
    // Scalar tail for k % kKPerStep != 0. The tail base can exceed k for the high lanes
    // (e.g. k=10240 with 128 lanes at full_k==k), so the whole tail is guarded.
    if (base < k) {
        const std::int64_t k0 = full_k + base;
#pragma unroll
        for (int r = 0; r < NRows; ++r) {
            const __nv_bfloat16* wr = weight + static_cast<std::int64_t>(r) * k + k0;
#pragma unroll
            for (int j = 0; j < kPack; ++j) {
                const std::int64_t kk = k0 + j;
                if (kk < k) {
                    acc[r] = fmaf(__bfloat162float(wr[j]), __bfloat162float(xtok[kk]), acc[r]);
                }
            }
        }
    }

    const int warp_id = lane / kWarpSize;
    const int warp_l  = lane % kWarpSize;
#pragma unroll
    for (int r = 0; r < NRows; ++r) {
        // The shuffle-based reduce needs the full converged warp; only lane 0 stores.
        const float warp_sum = warp_reduce_sum(acc[r]);
        if (warp_l == 0) { partials[warp_id][r] = warp_sum; }
    }
    __syncthreads();
    if (lane < NRows) {
        float total = 0.0F;
#pragma unroll
        for (int w = 0; w < kThreads / kWarpSize; ++w) {
            total += partials[w][lane];
        }
        // Token-major output: element (tok, r) is at tok*NRows + r.
        out[static_cast<std::int64_t>(tok) * NRows + lane] = __float2bfloat16_rn(total);
    }
}

} // namespace ninfer::ops::detail
