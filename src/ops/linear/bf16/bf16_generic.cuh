#pragma once

// Generic shape-generic BF16 x BF16 SIMT GEMM for the qwen4exp (jbq4) dense projections that fit
// no tuned exact-problem GEMV/MMA tile family.
//
//   out = weight * x
//
//   weight is contiguous BF16 [N, K] (column fastest, element (n, col) at n*K + col).
//   x      is contiguous BF16 [K, T] (dim0 fastest, element (col, t) at col + t*K).
//   out    is contiguous BF16 [N, T] (dim0 fastest, element (n, t) at n + t*N).
//
// One warp owns one output row; every lane accumulates the K dot-product slice for all T tokens
// (T <= kMaxTokens, held in registers) and reduces across the warp. K is streamed in 16-byte
// vectorized 8-element chunks with a scalar tail for k % (32*8). The kernel is correct for
// arbitrary N, K, and T in [1, kMaxTokens]; N/K/T are passed at runtime so a single instantiation
// serves every qwen4exp dense shape. The oracle (linear) evaluates the full dot product naively
// in FP64 from the represented inputs; the BF16 output is promoted and compared directly.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct alignas(16) Bf16GenericPack8 {
    std::uint32_t words[4];
};
static_assert(sizeof(Bf16GenericPack8) == 16);

template <int kMaxTokens, int kWarps>
__global__ void bf16_generic_gemm_kernel(const __nv_bfloat16* __restrict__ weight,
                                         const __nv_bfloat16* __restrict__ x,
                                         __nv_bfloat16* __restrict__ out, std::int32_t n,
                                         std::int32_t k, std::int32_t t) {
    constexpr int kPack      = 8;
    constexpr int kKPerStep  = kWarpSize * kPack;
    static_assert(kMaxTokens >= 1);

    const std::int32_t row    = static_cast<std::int32_t>(blockIdx.x) * kWarps +
                        (static_cast<std::int32_t>(threadIdx.x) / kWarpSize);
    const std::int32_t lane   = static_cast<int>(threadIdx.x % kWarpSize);
    if (row >= n) { return; }

    const std::int32_t col_base = lane * kPack;
    const __nv_bfloat16* wr     = weight + static_cast<std::int64_t>(row) * k + col_base;
    float acc[kMaxTokens];
#pragma unroll
    for (int i = 0; i < kMaxTokens; ++i) { acc[i] = 0.0F; }

    const std::int32_t full_k = (k / kKPerStep) * kKPerStep;
    for (std::int32_t i0 = 0; i0 < full_k; i0 += kKPerStep) {
        const std::int32_t k0 = i0 + col_base;
        const Bf16GenericPack8 wp = load_vec<Bf16GenericPack8>(wr + i0);
        float w[kPack];
#pragma unroll
        for (int j = 0; j < kPack / 2; ++j) {
            const float2 f = bf16x2_bits_to_float2(wp.words[j]);
            w[2 * j]       = f.x;
            w[2 * j + 1]   = f.y;
        }
#pragma unroll
        for (int tk = 0; tk < kMaxTokens; ++tk) {
            if (tk < t) {
                const __nv_bfloat16* xc = x + (static_cast<std::int64_t>(tk) * k) + k0;
                const Bf16GenericPack8 xp = load_vec<Bf16GenericPack8>(xc);
#pragma unroll
                for (int j = 0; j < kPack; ++j) {
                    const float xv =
                        __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&xp)[j]);
                    acc[tk] = fmaf(w[j], xv, acc[tk]);
                }
            }
        }
    }
    // Scalar tail for k % kKPerStep != 0. The tail base can exceed k for the high lanes, so the
    // whole tail is guarded at the k boundary.
    if (col_base < k) {
        const std::int32_t k0 = full_k + col_base;
        const __nv_bfloat16* wr = weight + static_cast<std::int64_t>(row) * k;
#pragma unroll
        for (int j = 0; j < kPack; ++j) {
            const std::int32_t kk = k0 + j;
            if (kk < k) {
                const float wv = __bfloat162float(wr[kk]);
#pragma unroll
                for (int tk = 0; tk < kMaxTokens; ++tk) {
                    if (tk < t) {
                        const float xv = __bfloat162float(x[static_cast<std::int64_t>(tk) * k + kk]);
                        acc[tk]        = fmaf(wv, xv, acc[tk]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int tk = 0; tk < kMaxTokens; ++tk) {
        if (tk < t && lane == 0) {
            out[static_cast<std::int64_t>(row) + static_cast<std::int64_t>(tk) * n] =
                __float2bfloat16_rn(warp_reduce_sum(acc[tk]));
        }
    }
}

} // namespace ninfer::ops::detail
