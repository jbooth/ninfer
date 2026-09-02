#pragma once

// Implements: include/ninfer/ops/qsa.h
//
// QSA sparse grouped-query attention (qwen4exp). One block per (head h, token t) gathers the
// indexer-selected live keys (<= kQsaTopKWidth) from the key-major dense k/v and runs an online
// (single-pass) softmax over the live set. Correctness-first v1: 256 threads (one per dim), one
// block reduction per visible key. Layouts (contiguous):
//   q/h/out : [256, query_heads, T]  element (d, h, t) at d*query_heads*T + h*T + t
//   k/v     : [n_kv, kv_heads, 256]   element (j, kvh, d) at (j*kv_heads + kvh)*256 + d
//   topk    : [2051, T]               element (slot, t) at slot*T + t   (-1 padded)
//   pos     : [T]                     absolute query position

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <limits>

namespace ninfer::ops {

namespace {

constexpr std::int32_t kQsaHeadDim  = 256;
constexpr std::int32_t kQsaTopKSlots = 2051;
constexpr std::int32_t kQsaThreads  = 256;

} // namespace

template <std::int32_t QueryHeads, std::int32_t KVHeads>
__global__ void qsa_flash_kernel(const __nv_bfloat16* __restrict__ q,
                                 const std::int32_t* __restrict__ topk,
                                 const std::int32_t* __restrict__ positions,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, std::int32_t n_kv,
                                 std::int32_t tokens, float scale, __nv_bfloat16* __restrict__ out) {
    constexpr std::int32_t group = QueryHeads / KVHeads;

    const std::int32_t h = blockIdx.y;
    const std::int32_t t = blockIdx.x;
    const std::int32_t d = threadIdx.x;
    const std::int32_t kvh = h / group;

    // q[d, h, t]
    const float qd = __bfloat162float(
        q[static_cast<std::int64_t>(d) * QueryHeads * tokens + static_cast<std::int64_t>(h) * tokens + t]);
    const std::int32_t pos = positions[t];

    float o = 0.0f;
    __shared__ float shared_m, shared_l, shared_s;
    __shared__ float warp_sums[kQsaThreads / 32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0) {
        shared_m = -INFINITY;
        shared_l = 0.0f;
    }
    __syncthreads();

    const std::int64_t topk_col = t;                       // topk[slot, t] = topk[slot*T + t]

    for (std::int32_t slot = 0; slot < kQsaTopKSlots; ++slot) {
        const std::int32_t j = topk[slot * tokens + topk_col];
        if (j < 0 || j >= n_kv || j > pos) continue;        // dead slot or outside the causal prefix

        const std::int64_t krow = (static_cast<std::int64_t>(j) * KVHeads + kvh) * kQsaHeadDim;
        const float kv = __bfloat162float(k[krow + d]);

        // Block reduction of qd * k over the 256 dims.
        float acc = qd * kv;
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_xor_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0) warp_sums[warp] = acc;
        __syncthreads();
        if (warp == 0) {
            float w = (lane < kQsaThreads / 32) ? warp_sums[lane] : 0.0f;
            for (int offset = 4; offset > 0; offset /= 2) {  // 8 warps
                w += __shfl_xor_sync(0xffu, w, offset);
            }
            if (lane == 0) shared_s = scale * w;
        }
        __syncthreads();

        const float s = shared_s;
        const float m_old = shared_m;
        const float m_new = fmaxf(m_old, s);
        const float alpha = expf(m_old - m_new);
        const float p     = expf(s - m_new);
        const std::int64_t vrow = (static_cast<std::int64_t>(j) * KVHeads + kvh) * kQsaHeadDim;
        const float vv  = __bfloat162float(v[vrow + d]);
        if (threadIdx.x == 0) {
            shared_m = m_new;
            shared_l = shared_l * alpha + p;
        }
        o = o * alpha + p * vv;
        __syncthreads();
    }

    const float l = shared_l;
    const float r = (l > 0.0f) ? o / l : 0.0f;
    out[static_cast<std::int64_t>(d) * QueryHeads * tokens + static_cast<std::int64_t>(h) * tokens + t] =
        __float2bfloat16_rn(r);
}

// ---------------------------------------------------------------------------
// q|gate de-interleave (kernel N).
//
// The attn_input_proj parent [13312, T] ships the rows in q | k | gate | v order (the converter
// de-interleaves the source per-head-fused attn_q at write time). Within the 6144-row q block and
// the 6144-row gate block, head h / dim d sits at row h*256 + d (head-major). qsa_softmax_attention
// wants q and gate dim-major [256, 24, T] (element (d, h, t) at d*24*T + h*T + t). This kernel
// performs that head-major -> dim-major transpose for the q and gate blocks in one pass; k/v are
// consumed by the dense-buffer append path, not here.
//   q[d, h, t]    = parent[0     + (h*256 + d)*T + t]
//   gate[d, h, t] = parent[6656  + (h*256 + d)*T + t]

constexpr std::int32_t kQsaParentRows    = 13312;
constexpr std::int32_t kQsaQOffset       = 0;     // q block: head h, dim d at row h*256 + d
constexpr std::int32_t kQsaGateOffset    = 6656;  // gate block (after q 6144 + k 512)
constexpr std::int32_t kQsaQHeadRows     = 256;   // q rows per head
constexpr std::int32_t kQsaQGateElements = 256 * 24;  // per output (dim-major)

__global__ void qsa_split_qgate_kernel(const __nv_bfloat16* __restrict__ parent,
                                       __nv_bfloat16* __restrict__ q,
                                       __nv_bfloat16* __restrict__ gate, std::int32_t tokens) {
    const std::int64_t total = static_cast<std::int64_t>(kQsaQGateElements) * tokens;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const std::int32_t t  = static_cast<std::int32_t>(i % tokens);
        const std::int64_t dh = i / tokens;             // dh = d*24 + h
        const std::int32_t h  = static_cast<std::int32_t>(dh % 24);
        const std::int32_t d  = static_cast<std::int32_t>(dh / 24);
        const std::int64_t row = static_cast<std::int64_t>(h) * kQsaQHeadRows + d;
        q[i]    = parent[(kQsaQOffset + row) * static_cast<std::int64_t>(tokens) + t];
        gate[i] = parent[(kQsaGateOffset + row) * static_cast<std::int64_t>(tokens) + t];
    }
}

} // namespace ninfer::ops
