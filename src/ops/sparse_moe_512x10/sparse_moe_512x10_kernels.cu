#include "ops/sparse_moe_512x10/sparse_moe_512x10_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops::sparse_moe_512x10_detail {
namespace {

constexpr float kNormWClamp = 6.1035e-5F;  // 2^-14 (llama.cpp expert gate norm floor)

__device__ __forceinline__ float bf16f(const __nv_bfloat16 v) { return __bfloat162float(v); }

// One warp computes one dot of a FP32-row x BF16-column reduction (k = K). K % 32 == 0.
template <int K>
__device__ __forceinline__ float router_dot(const float* __restrict__ wrow, int lane,
                                            const __nv_bfloat16* __restrict__ x, int T) {
    constexpr int kper = K / 32;
    float a = 0.0F;
    for (int i = 0; i < kper; ++i) {
        const int k = lane * kper + i;
        a = fmaf(wrow[k], bf16f(x[static_cast<std::int64_t>(k) * T]), a);
    }
    return warp_reduce_sum(a);
}

// One warp computes two NVFP4 dots (rows row0, row1) over k = K, writing (r0, r1). K % 16 == 0.
template <int K>
__device__ __forceinline__ void nvfp4_dot2(const std::uint8_t* __restrict__ codes,
                                           const std::uint8_t* __restrict__ scales, std::int64_t row0,
                                           std::int64_t row1, int lane,
                                           const __nv_bfloat16* __restrict__ x, int T, int t, float& r0,
                                           float& r1) {
    constexpr int kScales = K / 16;
    float a0 = 0.0F, a1 = 0.0F;
    for (int c = lane; c < kScales; c += 32) {
        const int k0       = c * 16;
        const std::uint8_t* cg0 = codes + row0 * (K / 2) + (k0 / 2);
        const std::uint8_t* cg1 = codes + row1 * (K / 2) + (k0 / 2);
        const float sc0 = detail::decode_nvfp4_e4m3(scales[row0 * (K / 16) + c]);
        const float sc1 = detail::decode_nvfp4_e4m3(scales[row1 * (K / 16) + c]);
        float pg = 0.0F, pu = 0.0F;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float x0 = bf16f(x[static_cast<std::int64_t>(k0 + 2 * i) * T + t]);
            const float x1 = bf16f(x[static_cast<std::int64_t>(k0 + 2 * i + 1) * T + t]);
            const float2 w0 = detail::decode_nvfp4_e2m1x2(cg0[i]);
            const float2 w1 = detail::decode_nvfp4_e2m1x2(cg1[i]);
            pg = fmaf(w0.x, x0, pg);
            pg = fmaf(w0.y, x1, pg);
            pu = fmaf(w1.x, x0, pu);
            pu = fmaf(w1.y, x1, pu);
        }
        a0 += pg * sc0;
        a1 += pu * sc1;
    }
    r0 = warp_reduce_sum(a0);
    r1 = warp_reduce_sum(a1);
}

// One warp computes one W8 dot (row) over k = K. K % 32 == 0.
template <int K>
__device__ __forceinline__ float w8_dot(const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales, std::int64_t row, int lane,
                                        const __nv_bfloat16* __restrict__ x, int T, int t) {
    constexpr int kGroups = K / 32;
    float a = 0.0F;
    for (int g = 0; g < kGroups; ++g) {
        float w0, w1;
        detail::W8ScalarDecodeAtom::load_pair(codes, nullptr, scales, row * kGroups + g, lane, w0, w1);
        const int k = g * 32 + lane * 2;
        a           = fmaf(w0, bf16f(x[static_cast<std::int64_t>(k) * T + t]), a);
        a           = fmaf(w1, bf16f(x[static_cast<std::int64_t>(k + 1) * T + t]), a);
    }
    return warp_reduce_sum(a);
}

// One warp computes one Q6 dot (row) over k = K. K % 64 == 0.
template <int K>
__device__ __forceinline__ float q6_dot(const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ high,
                                        const std::uint8_t* __restrict__ scales, std::int64_t row, int lane,
                                        const __nv_bfloat16* __restrict__ a, int T, int t) {
    constexpr int kGroups = K / 64;
    constexpr int kChunks = K / 8;
    float acc = 0.0F;
    for (int c = lane; c < kChunks; c += 32) {
        const int group = c / 8;
        const int sub   = c % 8;
        const std::int64_t g = row * kGroups + group;
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + g * detail::Q6RowSplitStorage::kCodeBytesPerGroup + sub * 4);
        const std::uint16_t hb = *reinterpret_cast<const std::uint16_t*>(
            high + g * detail::Q6RowSplitStorage::kHighBytesPerGroup + sub * 2);
        const std::uint16_t sb = *reinterpret_cast<const std::uint16_t*>(
            scales + g * detail::Q6RowSplitStorage::kScaleBytesPerGroup);
        float w[8];
        detail::Q6SimtDecodeAtom::decode_eight(packed, hb, sb, w);
        const int k0 = group * 64 + sub * 8;
        // decode_eight emits a swizzled order: weights[0..3] hold the even-k values
        // (k0, k0+2, k0+4, k0+6) and weights[4..7] the odd-k values (k0+1, k0+3, k0+5, k0+7).
        // Pair each weight with its own activation.
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int k = (v < 4) ? (k0 + 2 * v) : (k0 + 2 * (v - 4) + 1);
            acc = fmaf(w[v], bf16f(a[static_cast<std::int64_t>(k) * T + t]), acc);
        }
    }
    return warp_reduce_sum(acc);
}

// ---- K1: router GEMM (FP32 router @ BF16 x) -> scores[n_experts+1, T] ---------------
template <int K>
__global__ void router_kernel(const float* __restrict__ router, const __nv_bfloat16* __restrict__ x,
                              float* __restrict__ scores, std::int32_t rows, std::int32_t T) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_id >= rows * T) { return; }
    const int t   = warp_id % T;
    const int row = warp_id / T;
    const int lane = threadIdx.x & 31;
    scores[warp_id] = router_dot<K>(router + static_cast<std::int64_t>(row) * K, lane, x, t);
}

// ---- K2: softmax + top-k + norm_w (one 512-thread block per token) -------------------
// top_k must be <= 16. Selection is (prob desc, lower-id asc) and deterministic.
__global__ void topk_kernel(const float* __restrict__ scores, std::int32_t n_experts, std::int32_t top_k,
                            std::int32_t T, int* __restrict__ topk_ids, float* __restrict__ topk_w,
                            float* __restrict__ shared_logit) {
    constexpr int kThreads = 512;
    constexpr int kWarps   = kThreads / 32;  // 16
    __shared__ float prob[kThreads];
    __shared__ float red_v[kWarps];
    __shared__ int red_i[kWarps];
    __shared__ int sel_id[16];
    __shared__ float sel_w[16];
    const int t   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5, lane = tid & 31;
    // scores layout is [n_experts+1, T]: score for expert e, token t is scores[e*T + t].
    if (tid == 0) { shared_logit[t] = scores[static_cast<std::int64_t>(n_experts) * T + t]; }
    float p = (tid < n_experts) ? scores[static_cast<std::int64_t>(tid) * T + t] : -INFINITY;
    prob[tid] = p;
    __syncthreads();
    // block-max
    float m = p;
#pragma unroll
    for (int o = 1; o <= 16; o <<= 1) { m = fmaxf(m, __shfl_xor_sync(~0u, m, o)); }
    if (lane == 0) { red_v[warp] = m; }
    __syncthreads();
    if (tid < 32) {
        float v = (tid < kWarps) ? red_v[tid] : -INFINITY;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(~0u, v, o)); }
        if (tid == 0) { red_v[0] = v; }
    }
    __syncthreads();
    const float mx = red_v[0];
    // block-sum of exp
    float e = (tid < n_experts) ? expf(scores[static_cast<std::int64_t>(tid) * T + t] - mx) : 0.0F;
    prob[tid] = e;
    float s = e;
#pragma unroll
    for (int o = 1; o <= 16; o <<= 1) { s += __shfl_xor_sync(~0u, s, o); }
    if (lane == 0) { red_v[warp] = s; }
    __syncthreads();
    if (tid < 32) {
        float v = (tid < kWarps) ? red_v[tid] : 0.0F;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) { v += __shfl_xor_sync(~0u, v, o); }
        if (tid == 0) { red_v[0] = v; }
    }
    __syncthreads();
    const float sum = red_v[0];
    for (int i = tid; i < n_experts; i += kThreads) { prob[i] = expf(scores[static_cast<std::int64_t>(i) * T + t] - mx) / sum; }
    __syncthreads();
    // top-k: 10 block-argmax rounds (prob desc, lower-id asc)
    for (int round = 0; round < top_k; ++round) {
        float bv = prob[tid];
        int bi = tid;
        if (tid >= n_experts) { bv = -1.0F; }
#pragma unroll
        for (int o = 1; o <= 16; o <<= 1) {
            const float ov = __shfl_xor_sync(~0u, bv, o);
            const int oi   = __shfl_xor_sync(~0u, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
        __syncthreads();
        if (tid < 32) {
            float v  = (tid < kWarps) ? red_v[tid] : -1.0F;
            int id   = (tid < kWarps) ? red_i[tid] : 0x7fffffff;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(~0u, v, o);
                const int oi   = __shfl_xor_sync(~0u, id, o);
                if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
            }
            if (tid == 0) { red_v[0] = v; red_i[0] = id; }
        }
        __syncthreads();
        if (tid == 0) {
            sel_id[round] = static_cast<int>(red_i[0]);
            sel_w[round]  = red_v[0];
            const int winner = static_cast<int>(red_i[0]);
            if (winner < n_experts) { prob[winner] = -1.0F; }  // remove for the next round
        }
        __syncthreads();
    }
    if (tid == 0) {
        float wsum = 0.0F;
        for (int i = 0; i < top_k; ++i) {
            if (sel_id[i] < n_experts) { wsum += sel_w[i]; }
        }
        const float denom = fmaxf(wsum, kNormWClamp);
        for (int i = 0; i < top_k; ++i) {
            const int id = sel_id[i];
            topk_ids[i * T + t] = (id < n_experts) ? id : -1;
            topk_w[i * T + t]   = (id < n_experts) ? sel_w[i] / denom : 0.0F;
        }
    }
}

// ---- K3: grouped NVFP4 gate/up GEMM + silu_mul -> act[top_k, inter, T] ---------------
template <int K, int INTER>
__global__ void nvfp4_gate_up_kernel(const __nv_bfloat16* __restrict__ x,
                                     const int* __restrict__ topk_ids,
                                     const std::uint8_t* __restrict__ codes,
                                     const std::uint8_t* __restrict__ scales, std::int32_t top_k,
                                     std::int32_t T, __nv_bfloat16* __restrict__ act) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_id >= top_k * INTER * T) { return; }
    const int t   = warp_id % T;
    const int rem = warp_id / T;
    const int j   = rem % INTER;
    const int el  = rem / INTER;
    const int e   = topk_ids[el * T + t];
    if (e < 0) { return; }
    const int lane = threadIdx.x & 31;
    const std::int64_t gk = static_cast<std::int64_t>(e) * (2 * INTER);
    float gate = 0.0F, up = 0.0F;
    nvfp4_dot2<K>(codes, scales, gk + j, gk + INTER + j, lane, x, T, t, gate, up);
    if (lane == 0) { act[warp_id] = __float2bfloat16(silu(gate) * up); }
}

// ---- K4: grouped Q6 down GEMM -> y[top_k, hidden, T] (FP32) --------------------------
template <int INTER, int HIDDEN>
__global__ void q6_down_kernel(const __nv_bfloat16* __restrict__ act, const int* __restrict__ topk_ids,
                               const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
                               const std::uint8_t* __restrict__ scales, std::int32_t top_k, std::int32_t T,
                               float* __restrict__ y) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_id >= top_k * HIDDEN * T) { return; }
    const int t   = warp_id % T;
    const int rem = warp_id / T;
    const int i   = rem % HIDDEN;
    const int el  = rem / HIDDEN;
    const int e   = topk_ids[el * T + t];
    if (e < 0) { return; }
    const int lane = threadIdx.x & 31;
    const std::int64_t grow = static_cast<std::int64_t>(e) * HIDDEN + i;
    const __nv_bfloat16* arow = act + static_cast<std::int64_t>(el) * INTER * T;
    const float yval = q6_dot<INTER>(codes, high, scales, grow, lane, arow, T, t);
    if (lane == 0) { y[warp_id] = yval; }
}

// ---- K5: shared expert (gate/up + silu_mul + down) -> shared_y[hidden, T] ------------
template <int HIDDEN, int INTER>
__global__ void shared_gu_kernel(const __nv_bfloat16* __restrict__ x,
                                 const std::uint8_t* __restrict__ gu_codes,
                                 const std::uint8_t* __restrict__ gu_scales, std::int32_t T,
                                 __nv_bfloat16* __restrict__ shared_act) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_id >= INTER * T) { return; }
    const int t   = warp_id % T;
    const int j   = warp_id / T;
    const int lane = threadIdx.x & 31;
    float gate = 0.0F, up = 0.0F;
    // gate row = j, up row = INTER + j (W8, k=HIDDEN)
    gate = w8_dot<HIDDEN>(gu_codes, gu_scales, j, lane, x, T, t);
    up   = w8_dot<HIDDEN>(gu_codes, gu_scales, INTER + j, lane, x, T, t);
    if (lane == 0) { shared_act[warp_id] = __float2bfloat16(silu(gate) * up); }
}

template <int HIDDEN, int INTER>
__global__ void shared_down_kernel(const __nv_bfloat16* __restrict__ shared_act,
                                   const std::uint8_t* __restrict__ down_codes,
                                   const std::uint8_t* __restrict__ down_scales, std::int32_t T,
                                   float* __restrict__ shared_y) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_id >= HIDDEN * T) { return; }
    const int t = warp_id % T;
    const int i = warp_id / T;
    const int lane = threadIdx.x & 31;
    const float syval = w8_dot<INTER>(down_codes, down_scales, i, lane, shared_act, T, t);
    if (lane == 0) { shared_y[warp_id] = syval; }
}

// ---- K6: merge -> destination += Sigma w_e y_e + sigmoid(shared_logit) shared_y -------
__global__ void merge_kernel(const float* __restrict__ y, const float* __restrict__ shared_y,
                             const float* __restrict__ topk_w, const float* __restrict__ shared_logit,
                             std::int32_t top_k, std::int32_t hidden, std::int32_t T,
                             __nv_bfloat16* __restrict__ destination) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (warp_id >= hidden * T) { return; }
    const int t = warp_id % T;
    const int i = warp_id / T;
    const int lane = threadIdx.x & 31;
    float acc = 0.0F;
    for (int el = lane; el < top_k; el += 32) {
        acc = fmaf(topk_w[el * T + t], y[(static_cast<std::int64_t>(el) * hidden + i) * T + t], acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const float gated = sigmoid(shared_logit[t]) * shared_y[static_cast<std::int64_t>(i) * T + t];
        destination[warp_id] = __float2bfloat16(bf16f(destination[warp_id]) + acc + gated);
    }
}

} // namespace

void run(const __nv_bfloat16* x, std::int32_t n_experts, std::int32_t top_k, std::int32_t T,
         const float* router, float /*input_divisor*/, const std::uint8_t* gate_up_codes,
         const std::uint8_t* gate_up_scales, const std::uint8_t* down_codes,
         const std::uint8_t* down_high, const std::uint8_t* down_scales,
         const std::uint8_t* shared_gu_codes, const std::uint8_t* shared_gu_scales,
         const std::uint8_t* shared_down_codes, const std::uint8_t* shared_down_scales,
         __nv_bfloat16* destination, float* scores, int* topk_ids, float* topk_w,
         float* shared_logit, __nv_bfloat16* act, float* y, __nv_bfloat16* shared_act,
         float* shared_y, cudaStream_t stream) {
    constexpr int K = 2560, INTER = 640, HIDDEN = 2560;  // qwen4exp (compiled kernels)
    (void)router;  // used below
    {  // K1 router GEMM: (n_experts+1) x T warps
        const int rows = n_experts + 1;
        const int total = rows * T;
        const int blocks = (total + 7) / 8;
        router_kernel<K><<<blocks, 256, 0, stream>>>(router, x, scores, rows, T);
    }
    // K2 topk: T blocks of 512
    topk_kernel<<<T, 512, 0, stream>>>(scores, n_experts, top_k, T, topk_ids, topk_w, shared_logit);
    {  // K3 NVFP4 gate/up + silu_mul
        const int total = top_k * INTER * T;
        const int blocks = (total + 7) / 8;
        nvfp4_gate_up_kernel<K, INTER><<<blocks, 256, 0, stream>>>(
            x, topk_ids, gate_up_codes, gate_up_scales, top_k, T, act);
    }
    {  // K4 Q6 down
        const int total = top_k * HIDDEN * T;
        const int blocks = (total + 7) / 8;
        q6_down_kernel<INTER, HIDDEN><<<blocks, 256, 0, stream>>>(
            act, topk_ids, down_codes, down_high, down_scales, top_k, T, y);
    }
    {  // K5a shared gate/up
        const int total = INTER * T;
        const int blocks = (total + 7) / 8;
        shared_gu_kernel<HIDDEN, INTER><<<blocks, 256, 0, stream>>>(
            x, shared_gu_codes, shared_gu_scales, T, shared_act);
    }
    {  // K5b shared down
        const int total = HIDDEN * T;
        const int blocks = (total + 7) / 8;
        shared_down_kernel<HIDDEN, INTER><<<blocks, 256, 0, stream>>>(
            shared_act, shared_down_codes, shared_down_scales, T, shared_y);
    }
    {  // K6 merge
        const int total = HIDDEN * T;
        const int blocks = (total + 7) / 8;
        merge_kernel<<<blocks, 256, 0, stream>>>(
            y, shared_y, topk_w, shared_logit, top_k, HIDDEN, T, destination);
    }
}

} // namespace ninfer::ops::sparse_moe_512x10_detail
