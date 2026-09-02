#pragma once

// Implements: include/ninfer/ops/ple_mix.h
//
// qwen4exp personalization-layer (PLE) small kernels. Layouts are contiguous [rows, T] with the
// token dimension fastest-in-memory-per-column (element [i, t] at i * T + t), except the
// dilated-conv weight which is tap-major [4, C] (the stored artifact layout). v1 kernels are
// correctness-first: one block per (stream, token) for the gate reduction, one thread per
// element for the broadcast, and grid-stride elementwise routes for the conv.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// PLE gate: one block owns one (stream s, token t). Each thread accumulates the strided
// stream_dim-width dot (stride = tokens) in FP32, the block reduces, and thread 0 applies
// sgn(dot) * sqrt(clamp(|dot|, 1e-6, inf)) through the sigmoid.
__global__ void ple_stream_gate_kernel(const __nv_bfloat16* __restrict__ key,
                                       const __nv_bfloat16* __restrict__ query,
                                       float* __restrict__ out, std::int32_t stream_dim,
                                       std::int32_t tokens, float inv_sqrt_stream) {
    const std::int32_t t = blockIdx.x;
    const std::int32_t s = blockIdx.y;
    const std::int64_t base = static_cast<std::int64_t>(s) * stream_dim * tokens + t;

    float acc = 0.0f;
    for (std::int32_t i = threadIdx.x; i < stream_dim; i += 256) {
        const std::int64_t idx = base + static_cast<std::int64_t>(i) * tokens;
        acc += __bfloat162float(key[idx]) * __bfloat162float(query[idx]);
    }

    // Block reduce over 256 threads (8 warps).
    __shared__ float warp_sums[8];
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x / 32] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < 8; ++w) {
            total += warp_sums[w];
        }
        const float dot = total * inv_sqrt_stream;
        const float magnitude = fmaxf(fabsf(dot), 1.0e-6f);
        // sgn(dot) * sqrt(magnitude); copysignf carries -0 to -0 so sigmoid(-0) == 0.5.
        const float arg = copysignf(sqrtf(magnitude), dot);
        out[static_cast<std::int64_t>(s) * tokens + t] = 1.0f / (1.0f + expf(-arg));
    }
}

// PLE gated value broadcast: out[s*S + d, t] = value[d, t] * gate[s, t], one element per thread.
__launch_bounds__(256) __global__ void ple_gate_scale_kernel(const __nv_bfloat16* __restrict__ value,
                                                             const float* __restrict__ gate,
                                                             __nv_bfloat16* __restrict__ out,
                                                             std::int32_t stream_dim,
                                                             std::int32_t tokens, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t   = static_cast<std::int32_t>(i % tokens);
        const std::int64_t row = i / tokens;
        const std::int32_t s   = static_cast<std::int32_t>(row / stream_dim);
        const std::int32_t d   = static_cast<std::int32_t>(row % stream_dim);
        const float v = __bfloat162float(value[static_cast<std::int64_t>(d) * tokens + t]);
        const float g = gate[static_cast<std::int64_t>(s) * tokens + t];
        out[i] = __float2bfloat16_rn(v * g);
    }
}

// PLE dilated conv (kernel 4, dilation 3): out[c,t] = silu(sum_k w[k,C+c] * u[c,t-(3-k)*3]).
// One element per thread; the four taps read from the state (negative positions) or x.
__launch_bounds__(256) __global__ void ple_dilated_conv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, std::int32_t C,
    std::int32_t tokens, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i % tokens);
        const std::int64_t c = i / tokens;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const std::int32_t p = t - (3 - k) * 3; // -9, -6, -3, 0 relative to t
            const float u = (p >= 0)
                                ? __bfloat162float(x[c * tokens + p])
                                : __bfloat162float(state[c * 9 + (p + 9)]);
            acc += __bfloat162float(weight[k * static_cast<std::int64_t>(C) + c]) * u;
        }
        out[i] = __float2bfloat16_rn(silu(acc));
    }
}

// PLE dilated conv state update: new[c,j] = concat(state, x)[T + j], j in [0, 9). Pure copies of
// represented BF16 values (no arithmetic), so the result is bit-exact. One thread per channel
// reads the old window into registers before any store, which makes the exact in/out alias safe.
__launch_bounds__(256) __global__ void ple_dilated_conv_state_kernel(
    const __nv_bfloat16* __restrict__ state, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out_state, std::int32_t C, std::int32_t tokens) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t c = start; c < C; c += stride) {
        __nv_bfloat16 st[9];
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            st[j] = state[c * 9 + j];
        }
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            const std::int32_t p = tokens + j; // position in concat(state, x)
            out_state[c * 9 + j] = (p < 9) ? st[p] : x[c * tokens + (p - 9)];
        }
    }
}

} // namespace ninfer::ops
