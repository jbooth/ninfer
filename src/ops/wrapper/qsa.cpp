// ninfer::ops - QSA sparse attention wrapper: public API validation and launch dispatch.
#include "ninfer/ops/qsa.h"

#include "ops/launcher/qsa.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim      = 256;
constexpr std::int32_t kQueryHeads   = 24;
constexpr std::int32_t kKVHeads      = 2;

void require_qsa_geometry(AttentionHeadGeometry g) {
    if (g.head_dim != kHeadDim || g.query_heads != kQueryHeads || g.kv_heads != kKVHeads) {
        throw std::invalid_argument("qsa_softmax_attention: geometry must be [256,24,2]");
    }
}

std::int32_t require_q_out(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16 || t.ne[0] != kHeadDim || t.ne[1] != kQueryHeads || t.ne[2] < 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("qsa_softmax_attention: ") + label +
                                    " must be contiguous BF16 [256, 24, T] (dim, head, token)");
    }
    return static_cast<std::int32_t>(t.ne[2]);
}

void require_kv(const Tensor& t, std::int32_t n_kv, const char* label) {
    if (t.dtype != DType::BF16 || t.ne[0] != n_kv || t.ne[1] != kKVHeads || t.ne[2] != kHeadDim ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("qsa_softmax_attention: ") + label +
                                    " must be contiguous BF16 [n_kv, 2, 256]");
    }
}
} // namespace

void qsa_softmax_attention(const Tensor& q, const Tensor& topk_idx, const Tensor& positions,
                           const Tensor& k, const Tensor& v, std::int32_t n_kv,
                           AttentionHeadGeometry geometry, float scale, WorkspaceArena& workspace,
                           Tensor& out, cudaStream_t stream) {
    require_qsa_geometry(geometry);
    if (!std::isfinite(scale) || scale <= 0.0F) {
        throw std::invalid_argument("qsa_softmax_attention: scale must be finite and positive");
    }
    if (n_kv < 1) {
        throw std::invalid_argument("qsa_softmax_attention: n_kv must be >= 1");
    }
    const std::int32_t tokens = require_q_out(q, "q");
    require_q_out(out, "out");
    if (out.ne[2] != tokens) {
        throw std::invalid_argument("qsa_softmax_attention: out tokens must match q");
    }
    if (topk_idx.dtype != DType::I32 || topk_idx.ne[0] != kQsaTopKWidth || topk_idx.ne[1] != tokens ||
        topk_idx.ne[2] != 1 || topk_idx.ne[3] != 1 || !topk_idx.is_contiguous() ||
        topk_idx.data == nullptr) {
        throw std::invalid_argument("qsa_softmax_attention: topk_idx must be I32 [2051, T]");
    }
    if (positions.dtype != DType::I32 || positions.ne[0] != tokens || positions.ne[1] != 1 ||
        positions.ne[2] != 1 || positions.ne[3] != 1 || !positions.is_contiguous() ||
        positions.data == nullptr) {
        throw std::invalid_argument("qsa_softmax_attention: positions must be I32 [T]");
    }
    require_kv(k, n_kv, "k");
    require_kv(v, n_kv, "v");
    detail::qsa_softmax_attention_launch(q, topk_idx, positions, k, v, n_kv, scale, out, stream);
}

std::size_t qsa_softmax_attention_workspace_capacity_bytes(AttentionHeadGeometry geometry,
                                                           std::int32_t min_tokens,
                                                           std::int32_t max_tokens) {
    require_qsa_geometry(geometry);
    if (min_tokens < 1 || max_tokens < min_tokens) {
        throw std::invalid_argument("qsa_softmax_attention: invalid token interval");
    }
    return 0;  // the op uses only per-block shared memory; no caller-owned global scratch.
}

namespace {

constexpr std::int32_t kQsaParentRows = 13312;

std::int32_t require_split_parent(const Tensor& t) {
    if (t.dtype != DType::BF16 || t.ne[0] != kQsaParentRows || t.ne[1] < 1 || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(
            "qsa_split_qgate: parent must be contiguous BF16 [13312, T]");
    }
    return static_cast<std::int32_t>(t.ne[1]);
}

void require_split_out(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16 || t.ne[0] != kHeadDim || t.ne[1] != kQueryHeads || t.ne[2] < 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("qsa_split_qgate: ") + label +
                                    " must be contiguous BF16 [256, 24, T] (dim, head, token)");
    }
}
} // namespace

void qsa_split_qgate(const Tensor& parent, Tensor& q, Tensor& gate, cudaStream_t stream) {
    const std::int32_t tokens = require_split_parent(parent);
    require_split_out(q, "q");
    require_split_out(gate, "gate");
    if (q.ne[2] != tokens || gate.ne[2] != tokens) {
        throw std::invalid_argument("qsa_split_qgate: q and gate tokens must match parent");
    }
    detail::qsa_split_qgate_launch(parent, q, gate, stream);
}

namespace {

constexpr std::int32_t kIdxDim    = 128;
constexpr std::int32_t kIdxQProj  = 512;
constexpr std::int32_t kCompressR = 4;

std::int32_t require_idx_k_cache(const Tensor& t) {
    if (t.dtype != DType::BF16 || t.ne[0] != kIdxDim || t.ne[1] < 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument("qsa_indexer: k_cache must be contiguous BF16 [128, n_kv]");
    }
    return static_cast<std::int32_t>(t.ne[1]);
}

std::int32_t require_idx_q_raw(const Tensor& t) {
    if (t.dtype != DType::BF16 || t.ne[0] != kIdxQProj || t.ne[1] < 1 || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument("qsa_indexer: q_raw must be contiguous BF16 [512, T]");
    }
    return static_cast<std::int32_t>(t.ne[1]);
}

void require_idx_gamma(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16 || t.ne[0] != kIdxDim || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("qsa_indexer: ") + label +
                                    " must be contiguous BF16 [128]");
    }
}
} // namespace

std::size_t qsa_indexer_workspace_capacity_bytes(std::int32_t n_kv, std::int32_t T) {
    if (n_kv < 1 || T < 1) {
        throw std::invalid_argument("qsa_indexer: n_kv and T must be >= 1");
    }
    const std::size_t n_blocks = (static_cast<std::size_t>(n_kv) + kCompressR - 1) / kCompressR;
    const std::size_t pooled   = (static_cast<std::size_t>(kIdxDim) * n_blocks * 2 + 255u) & ~255ull;
    const std::size_t qout     = (static_cast<std::size_t>(kIdxQProj) * T * 2 + 255u) & ~255ull;
    return pooled + qout;
}

void qsa_indexer(const Tensor& k_cache, const Tensor& q_raw, const Tensor& positions,
                 const Tensor& k_norm, const Tensor& q_norm, Tensor& topk, Tensor& score,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t n_kv = require_idx_k_cache(k_cache);
    const std::int32_t T    = require_idx_q_raw(q_raw);
    if (positions.dtype != DType::I32 || positions.ne[0] != T || positions.ne[1] != 1 ||
        positions.ne[2] != 1 || positions.ne[3] != 1 || !positions.is_contiguous() ||
        positions.data == nullptr) {
        throw std::invalid_argument("qsa_indexer: positions must be contiguous I32 [T]");
    }
    require_idx_gamma(k_norm, "k_norm");
    require_idx_gamma(q_norm, "q_norm");
    if (topk.dtype != DType::I32 || topk.ne[0] != kQsaTopKWidth || topk.ne[1] != T || topk.ne[2] != 1 ||
        topk.ne[3] != 1 || !topk.is_contiguous() || topk.data == nullptr) {
        throw std::invalid_argument("qsa_indexer: topk must be contiguous I32 [2051, T]");
    }
    const std::int32_t n_blocks = (n_kv + kCompressR - 1) / kCompressR;
    if (score.dtype != DType::FP32 || score.ne[0] != n_blocks || score.ne[1] != T || score.ne[2] != 1 ||
        score.ne[3] != 1 || !score.is_contiguous() || score.data == nullptr) {
        throw std::invalid_argument("qsa_indexer: score must be contiguous FP32 [n_blocks, T]");
    }

    const std::size_t pooled_bytes = static_cast<std::size_t>(kIdxDim) * n_blocks * 2;
    const std::size_t qout_bytes   = static_cast<std::size_t>(kIdxQProj) * T * 2;
    DeviceSpan pooled_sp = workspace.alloc_bytes(pooled_bytes);
    DeviceSpan qout_sp   = workspace.alloc_bytes(qout_bytes);
    if (pooled_sp.data == nullptr || qout_sp.data == nullptr) {
        throw std::runtime_error("qsa_indexer: workspace allocation failed");
    }
    Tensor pooled_out(pooled_sp.data, DType::BF16, {kIdxDim, n_blocks});
    Tensor q_out(qout_sp.data, DType::BF16, {kIdxQProj, T});

    detail::qsa_indexer_launch(k_cache, q_raw, positions, k_norm, q_norm, pooled_out, q_out, topk,
                               score, stream);
}

} // namespace ninfer::ops
