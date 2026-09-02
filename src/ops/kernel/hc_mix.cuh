#pragma once

// Implements: include/ninfer/ops/hc_mix.h
//
// qwen4exp hyper-connection small kernels. All layouts are contiguous [rows, T] with the token
// dimension fastest-in-memory-per-column (element [i, t] at i * T + t). v1 kernels are
// correctness-first: one thread per (stream, token) for the grouped reduction, one thread per
// element for the elementwise routes; the decode hot path (T = 1) reads contiguous rows.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// K1: grouped RMSNorm. One thread owns one (stream s, token t) pair and reduces the
// stream_dim-width strided row (stride = tokens). gamma is the folded FP32 (1 + w).
__global__ void grouped_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                       const float* __restrict__ gamma,
                                       __nv_bfloat16* __restrict__ out, std::int32_t stream_dim,
                                       std::int32_t tokens, float eps) {
    const std::int32_t t = blockIdx.x;
    const std::int32_t s = blockIdx.y;
    const std::int64_t row_base = static_cast<std::int64_t>(s) * stream_dim * tokens + t;

    float sum = 0.0f;
    for (std::int32_t i = 0; i < stream_dim; ++i) {
        const float v = __bfloat162float(x[row_base + static_cast<std::int64_t>(i) * tokens]);
        sum += v * v;
    }
    const float inv = rsqrtf(sum / static_cast<float>(stream_dim) + eps);

    for (std::int32_t i = 0; i < stream_dim; ++i) {
        const std::int64_t idx = row_base + static_cast<std::int64_t>(i) * tokens;
        const std::int32_t col = s * stream_dim + i;
        const float v = __bfloat162float(x[idx]);
        out[idx] = __float2bfloat16_rn(v * inv * gamma[col]);
    }
}

// K2: out = silu(x / div), one element per thread.
__launch_bounds__(256) __global__ void hc_silu_div4_kernel(const __nv_bfloat16* __restrict__ x,
                                                           __nv_bfloat16* __restrict__ out,
                                                           float div, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float v = __bfloat162float(x[i]) / div;
        out[i]        = __float2bfloat16_rn(silu(v));
    }
}

// K2: out[d, t] = (1/hc) * sum_s xn[s*sd + d, t] * sigmoid(gate[s*sd + d, t]).
// One thread owns one (d, t); the hc terms are tiny (hc = 4 in qwen4exp).
__global__ void hc_gate_mul_mean4_kernel(const __nv_bfloat16* __restrict__ xn,
                                         const __nv_bfloat16* __restrict__ gate,
                                         __nv_bfloat16* __restrict__ out, std::int32_t stream_dim,
                                         std::int32_t tokens, std::int32_t hc) {
    const std::int32_t d = blockIdx.x;
    const std::int32_t t = blockIdx.y;
    float acc = 0.0f;
    for (std::int32_t s = 0; s < hc; ++s) {
        const std::int64_t idx = (static_cast<std::int64_t>(s) * stream_dim + d) * tokens + t;
        const float x = __bfloat162float(xn[idx]);
        const float g = __bfloat162float(gate[idx]);
        acc += x * sigmoid(g);
    }
    out[static_cast<std::int64_t>(d) * tokens + t] =
        __float2bfloat16_rn(acc / static_cast<float>(hc));
}

// K2: w[s, t] = 2 * sigmoid(inject[s, t] / hc); out[s*sd + d, t] = res[s*sd+d, t] + block[d,t] w.
// One element per thread; the block term is a stream-strided broadcast.
__launch_bounds__(256) __global__ void hc_combine_kernel(const __nv_bfloat16* __restrict__ res,
                                                         const __nv_bfloat16* __restrict__ block,
                                                         const __nv_bfloat16* __restrict__ inject,
                                                         __nv_bfloat16* __restrict__ out,
                                                         std::int32_t stream_dim, std::int32_t hc,
                                                         std::int32_t tokens) {
    const std::int32_t total = hc * stream_dim * tokens;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < total; i += stride) {
        const std::int32_t t   = static_cast<std::int32_t>(i % tokens);
        const std::int32_t idx = static_cast<std::int32_t>(i / tokens); // [0, hc*stream_dim)
        const std::int32_t s   = idx / stream_dim;
        const std::int32_t d   = idx % stream_dim;
        const float w =
            2.0f * sigmoid(__bfloat162float(inject[static_cast<std::int64_t>(s) * tokens + t]) /
                           static_cast<float>(hc));
        const float r = __bfloat162float(res[i]);
        const float b = __bfloat162float(block[static_cast<std::int64_t>(d) * tokens + t]);
        out[i] = __float2bfloat16_rn(r + b * w);
    }
}

} // namespace ninfer::ops
