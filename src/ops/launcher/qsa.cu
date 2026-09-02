// ninfer::ops - QSA sparse attention launcher (single registered profile [256,24,2]).
#include "ops/launcher/qsa.h"

#include "ops/kernel/qsa.cuh"
#include "ops/kernel/qsa_indexer.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

void qsa_softmax_attention_launch(const Tensor& q, const Tensor& topk_idx, const Tensor& positions,
                                  const Tensor& k, const Tensor& v, std::int32_t n_kv,
                                  float scale, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kQueryHeads = 24;
    constexpr std::int32_t kKVHeads    = 2;
    const std::int32_t tokens = out.ne[2];
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(kQueryHeads), 1u);
    qsa_flash_kernel<kQueryHeads, kKVHeads><<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::int32_t*>(topk_idx.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        n_kv, tokens, scale, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void qsa_split_qgate_launch(const Tensor& parent, Tensor& q, Tensor& gate, cudaStream_t stream) {
    constexpr std::int32_t kThreads = 256;
    const std::int32_t tokens = q.ne[2];
    const std::int64_t total  = static_cast<std::int64_t>(kQsaQGateElements) * tokens;
    const std::int64_t exact  = (total + kThreads - 1) / kThreads;
    const unsigned blocks = static_cast<unsigned>(std::min<std::int64_t>(exact, 2048));
    qsa_split_qgate_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(parent.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(gate.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_indexer_launch(const Tensor& k_cache, const Tensor& q_raw, const Tensor& positions,
                        const Tensor& k_norm, const Tensor& q_norm,
                        Tensor& pooled_out, Tensor& q_out, Tensor& topk, Tensor& score,
                        cudaStream_t stream) {
    namespace D = ninfer::ops::qsa_idx_detail;
    const std::int32_t n_kv     = static_cast<std::int32_t>(k_cache.ne[1]);
    const std::int32_t T        = static_cast<std::int32_t>(positions.ne[0]);
    const std::int32_t n_blocks = (n_kv + D::kR - 1) / D::kR;

    const __nv_bfloat16* kc = static_cast<const __nv_bfloat16*>(k_cache.data);
    const __nv_bfloat16* kr = static_cast<const __nv_bfloat16*>(q_raw.data);
    const std::int32_t* pos = static_cast<const std::int32_t*>(positions.data);
    const __nv_bfloat16* kn = static_cast<const __nv_bfloat16*>(k_norm.data);
    const __nv_bfloat16* qn = static_cast<const __nv_bfloat16*>(q_norm.data);
    __nv_bfloat16* po       = static_cast<__nv_bfloat16*>(pooled_out.data);
    __nv_bfloat16* qo       = static_cast<__nv_bfloat16*>(q_out.data);
    std::int32_t* tp        = static_cast<std::int32_t*>(topk.data);
    float* sc               = static_cast<float*>(score.data);

    // 1: pool + rmsnorm + rope -> pooled_out [128, n_blocks] (one block per block).
    if (n_blocks > 0) {
        D::qsa_idx_pool_norm_rope<<<n_blocks, D::kDim, 0, stream>>>(kc, kn, po, n_kv, n_blocks);
    }
    // 2: per-(head, token) rmsnorm + rope -> q_out [512, T] (grid T x 4 heads).
    if (T > 0) {
        dim3 grid(T, D::kHeads);
        D::qsa_idx_q_norm_rope<<<grid, D::kDim, 0, stream>>>(kr, qn, pos, qo, T);
    }
    // 3: rectified 4-head score -> score [n_blocks, T] (grid n_blocks x T).
    if (n_blocks > 0 && T > 0) {
        dim3 grid(n_blocks, T);
        D::qsa_idx_score<<<grid, D::kDim, 0, stream>>>(po, qo, sc, n_blocks, T);
    }
    // 4: expand + deterministic top-k -> topk [2051, T] (one block per token).
    if (T > 0) {
        D::qsa_idx_topk<<<T, 512, 0, stream>>>(sc, pos, tp, n_kv, n_blocks, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
