#pragma once

// Implements: include/ninfer/ops/qsa.h::qsa_indexer (K5, qwen3.8-flash-next QSA indexer).
//
// The op owns the whole indexer pipeline over the dense side cache: block pooling of the raw
// cached keys, per-block RMSNorm + rope (block positions b*r), per-(head,token) RMSNorm + rope of
// the query projection (query positions), the 4-head rectified score, the causal / force-in
// per-cell expansion, and the deterministic top-k (lower-id tiebreak, ascending-id output).
//
// Reference: llama.cpp qwen4exp.cpp build_qsa_top_k (:477) + set_input_qsa
// (llama-memory-hybrid-idx.cpp:342). Single-stream dense text (cell j at position j).
//
// Geometry (exact, qwen3.8-flash-next):
//   idx_dim = 128 (indexer key length), n_idx_h = 4 (indexer heads), r = 4 (compress ratio),
//   top-k width = min(n_kv, 2051).  The q projection is [512, 2560] = [128, 4, T], head h dim k
//   at row 128h + k.  RMS eps = 1e-6.  rope: base 1e7 over the first 64 dims (32 pairs, split
//   rotate-half pairing (p, p+32)); the M-RoPE sections [11,11,10,0] collapse to a single position
//   for text, so it is standard rope.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

// 1e7-base inverse frequencies for the 32 rotated dim-pairs (identical to the TextMrope table in
// rope.cuh); pair p has frequency 1e7^(-2p/64) = 1e7^(-p/32).
static __device__ __constant__ float kQsaIdxInvFrequency[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

namespace qsa_idx_detail {

inline constexpr std::int32_t kDim    = 128;  // indexer key length
inline constexpr std::int32_t kHeads  = 4;    // indexer query heads
inline constexpr std::int32_t kQProj  = kDim * kHeads;  // 512
inline constexpr std::int32_t kR      = 4;    // compress ratio (block size)
inline constexpr std::int32_t kTopK   = 2051; // top-k width (indexer_top_k + r - 1)
inline constexpr std::int32_t kHalf   = 32;   // rotary_dim/2 (64 rotated dims)
inline constexpr float kRmsEps        = 1e-6F;

// Orderable uint32 key for a float: higher value -> higher key (the standard float->sortable map).
__device__ __forceinline__ std::uint32_t orderable_key(float v) {
    const std::uint32_t u = static_cast<std::uint32_t>(__float_as_int(v));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Expanded per-cell score for cell j (position j, dense text) and query token t at position q.
// Mirrors set_input_qsa's per-cell bias path: future cells -inf; the query's own (tail) block is
// force-in at +1e9; cells in complete earlier blocks carry the block score; cells in an
// incomplete non-tail block are -inf (unpooled).
__device__ __forceinline__ float expanded_value(const float* score, std::int32_t n_kv,
                                                std::int32_t j, std::int32_t t, std::int32_t q,
                                                std::int32_t n_blocks, std::int32_t T) {
    const std::int32_t tail_start = ((q + 1) / kR) * kR;
    if (j > q) {
        return -INFINITY;
    }
    if (j >= tail_start) {
        return 1e9F;  // the query's own (possibly incomplete) block is always visible
    }
    const std::int32_t b = j / kR;
    const bool complete = (b < n_blocks) && (b * kR + kR <= n_kv);
    return complete ? score[b * T + t] : -INFINITY;
}

// Shared-memory sum reduction over blockDim.x threads; the result is broadcast to s_dst.
template <unsigned Block>
__device__ __forceinline__ void block_reduce_sum(int value, int& s_dst) {
    __shared__ int s_part[Block / 32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wid  = static_cast<int>(threadIdx.x) >> 5;
    int v = value;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) {
        s_part[wid] = v;
    }
    __syncthreads();
    const int n_warp = Block / 32;
    if (wid == 0) {
        v = (lane < n_warp) ? s_part[lane] : 0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) {
            s_dst = v;
        }
    }
    __syncthreads();
}

// Pool block b's members (mean of r members, cell-0 padded for the tail), RMSNorm by k_norm, and
// rope at position b*r.  One block per block; 128 threads (one per dim).
__global__ void qsa_idx_pool_norm_rope(const __nv_bfloat16* __restrict__ k_cache,
                                       const __nv_bfloat16* __restrict__ k_norm,
                                       __nv_bfloat16* __restrict__ pooled_out, std::int32_t n_kv,
                                       std::int32_t n_blocks) {
    const std::int32_t b = static_cast<std::int32_t>(blockIdx.x);
    if (b >= n_blocks) {
        return;
    }
    const std::int32_t k = static_cast<std::int32_t>(threadIdx.x);
    float acc = 0.0F;
#pragma unroll
    for (std::int32_t o = 0; o < kR; ++o) {
        const std::int32_t j = b * kR + o;
        const std::int32_t jidx = (j < n_kv) ? j : 0;  // cell-0 padding for the tail block
        acc += __bfloat162float(k_cache[k * n_kv + jidx]);
    }
    const float pooled = acc / static_cast<float>(kR);
    __shared__ float s_pooled[kDim];
    s_pooled[k] = pooled;
    __syncthreads();
    float ss = 0.0F;
#pragma unroll
    for (std::int32_t i = 0; i < kDim; ++i) {
        ss += s_pooled[i] * s_pooled[i];
    }
    const float inv = rsqrtf(ss / static_cast<float>(kDim) + kRmsEps);
    const float normed = s_pooled[k] * inv * __bfloat162float(k_norm[k]);
    __shared__ float s_normed[kDim];
    s_normed[k] = normed;
    __syncthreads();
    if (k < kHalf) {
        const float first  = s_normed[k];
        const float second = s_normed[k + kHalf];
        float s, c;
        sincosf(static_cast<float>(b * kR) * kQsaIdxInvFrequency[k], &s, &c);
        pooled_out[k * n_blocks + b]              = __float2bfloat16_rn(first * c - second * s);
        pooled_out[(k + kHalf) * n_blocks + b]    = __float2bfloat16_rn(second * c + first * s);
    } else {
        pooled_out[k * n_blocks + b] = __float2bfloat16_rn(s_normed[k]);
    }
}

// Per-(head, token) RMSNorm by q_norm and rope at the query position.  q_raw is [512, T] (head h
// dim k at row 128h + k); one block per (head, token); 128 threads.
__global__ void qsa_idx_q_norm_rope(const __nv_bfloat16* __restrict__ q_raw,
                                    const __nv_bfloat16* __restrict__ q_norm,
                                    const std::int32_t* __restrict__ positions,
                                    __nv_bfloat16* __restrict__ q_out, std::int32_t T) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t h = static_cast<std::int32_t>(blockIdx.y);
    if (t >= T) {
        return;
    }
    const std::int32_t k = static_cast<std::int32_t>(threadIdx.x);
    const float x = __bfloat162float(q_raw[(h * kDim + k) * T + t]);
    __shared__ float s_x[kDim];
    s_x[k] = x;
    __syncthreads();
    float ss = 0.0F;
#pragma unroll
    for (std::int32_t i = 0; i < kDim; ++i) {
        ss += s_x[i] * s_x[i];
    }
    const float inv = rsqrtf(ss / static_cast<float>(kDim) + kRmsEps);
    const float normed = s_x[k] * inv * __bfloat162float(q_norm[k]);
    __shared__ float s_normed[kDim];
    s_normed[k] = normed;
    __syncthreads();
    if (k < kHalf) {
        const float first  = s_normed[k];
        const float second = s_normed[k + kHalf];
        float s, c;
        sincosf(static_cast<float>(positions[t]) * kQsaIdxInvFrequency[k], &s, &c);
        q_out[(h * kDim + k) * T + t]              = __float2bfloat16_rn(first * c - second * s);
        q_out[(h * kDim + k + kHalf) * T + t]      = __float2bfloat16_rn(second * c + first * s);
    } else {
        q_out[(h * kDim + k) * T + t] = __float2bfloat16_rn(s_normed[k]);
    }
}

// Rectified 4-head score: score[b, t] = sum_h relu( dot(pooled_out[:, b], q_out[h*128:(h+1)*128, t]) ).
// One block per (block, token); 128 threads.
__global__ void qsa_idx_score(const __nv_bfloat16* __restrict__ pooled_out,
                              const __nv_bfloat16* __restrict__ q_out, float* __restrict__ score,
                              std::int32_t n_blocks, std::int32_t T) {
    const std::int32_t b = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.y);
    if (b >= n_blocks || t >= T) {
        return;
    }
    const std::int32_t k = static_cast<std::int32_t>(threadIdx.x);
    __shared__ float s_part[kDim];
    __shared__ float s_dot[kHeads];
    const float pk = __bfloat162float(pooled_out[k * n_blocks + b]);
#pragma unroll
    for (std::int32_t h = 0; h < kHeads; ++h) {
        s_part[k] = pk * __bfloat162float(q_out[(h * kDim + k) * T + t]);
        __syncthreads();
        float sum = 0.0F;
#pragma unroll
        for (std::int32_t i = 0; i < kDim; ++i) {
            sum += s_part[i];
        }
        if (k == 0) {
            s_dot[h] = sum;
        }
        __syncthreads();
    }
    if (k == 0) {
        float acc = 0.0F;
#pragma unroll
        for (std::int32_t h = 0; h < kHeads; ++h) {
            acc += fmaxf(s_dot[h], 0.0F);
        }
        score[b * T + t] = acc;
    }
}

// Expand + deterministic top-k.  One block per token; 512 threads.  Finds the width-th largest
// orderable key by 8-pass radix select, counts keys strictly above it, then a sequential
// ascending-id scan collects all keys above the threshold plus the smallest `width - count_gt`
// keys equal to it, writing them in ascending cell order and padding the rest with -1.
__global__ void qsa_idx_topk(const float* __restrict__ score, const std::int32_t* __restrict__ positions,
                             std::int32_t* __restrict__ topk_idx, std::int32_t n_kv,
                             std::int32_t n_blocks, std::int32_t T) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x);
    if (t >= T) {
        return;
    }
    const std::int32_t q = positions[t];
    const std::int32_t width = (n_kv < kTopK) ? n_kv : kTopK;

    __shared__ int s_scalar;
    std::uint32_t result = 0;
    int k = width;
    for (std::int32_t bit = 31; bit >= 0; --bit) {
        const std::uint32_t mask = 1u << bit;
        const std::uint32_t above_mask =
            (bit + 1 >= 32) ? 0u : (0xFFFFFFFFu << (bit + 1));
        int local = 0;
        for (std::int32_t j = static_cast<std::int32_t>(threadIdx.x); j < n_kv;
             j += static_cast<std::int32_t>(blockDim.x)) {
            const std::uint32_t key = orderable_key(expanded_value(score, n_kv, j, t, q, n_blocks, T));
            if ((key & above_mask) == (result & above_mask) && (key & mask)) {
                ++local;
            }
        }
        block_reduce_sum<512>(local, s_scalar);
        const int count = s_scalar;
        if (count >= k) {
            result |= mask;
        } else {
            k -= count;
        }
    }

    int local_gt = 0;
    for (std::int32_t j = static_cast<std::int32_t>(threadIdx.x); j < n_kv;
         j += static_cast<std::int32_t>(blockDim.x)) {
        if (orderable_key(expanded_value(score, n_kv, j, t, q, n_blocks, T)) > result) {
            ++local_gt;
        }
    }
    block_reduce_sum<512>(local_gt, s_scalar);
    const int need = width - s_scalar;  // keys equal to `result` to take (smallest ids)

    // Ascending-id collection (sequential on thread 0; the per-token selection set).
    if (threadIdx.x == 0) {
        int seen_eq = 0;
        int slot = 0;
        for (std::int32_t j = 0; j < n_kv; ++j) {
            const std::uint32_t key =
                orderable_key(expanded_value(score, n_kv, j, t, q, n_blocks, T));
            if (key > result || (key == result && seen_eq < need)) {
                topk_idx[slot * T + t] = j;
                ++slot;
                if (key == result) {
                    ++seen_eq;
                }
            }
        }
        for (; slot < kTopK; ++slot) {
            topk_idx[slot * T + t] = -1;
        }
    }
}

} // namespace qsa_idx_detail

} // namespace ninfer::ops
