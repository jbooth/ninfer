#pragma once

#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Fixed QSA indexer top-k width (2048 cells + one compress block - 1). K5 emits a width-2051
// row per query token with -1 padding for slots beyond the live key count; K6 gathers exactly
// these ids.
inline constexpr std::int32_t kQsaTopKWidth = 2051;

/**
 * QSA sparse grouped-query attention: attend over the indexer-selected key set (qwen4exp query
 * sparse attention). For query head h (0-based) and token t, let kvh = floor(h / (query_heads /
 * kv_heads)) be the KV head it reads, and let J be the set of live ids in `topk_idx` for token t,
 * restricted to the causal prefix (j <= positions[t] and j < n_kv):
 *
 *   score[j]       = scale * sum_d FP64(q[h, d, t]) * FP64(k[j, kvh, d])
 *   probability[j] = exp(score[j] - max(J' score)) / sum_{x in J'} exp(score[x] - max(J' score))
 *   out[h, d, t]   = sum_{j in J'} probability[j] * FP64(v[j, kvh, d])
 *
 * where J' is the live set and max is taken over J' (empty J' yields exact zero output). The
 * complete formula is evaluated naively in FP64 by the oracle; dot products, the stable Softmax,
 * and the value reduction follow the shared attention oracle in `softmax_attention.h`. The BF16
 * output is promoted and compared directly; storage rounding belongs to the Op criterion.
 *
 * q and out are contiguous BF16 [256, query_heads, T] (dim, head, token; element (d, h, t) at
 * d*query_heads*T + h*T + t) - the rope-family Q layout, so the op composes directly with per-head
 * norm/rope and the [6144,T] output projection without transposes. k and v are contiguous BF16
 * [n_kv, kv_heads, 256] (key-major, then KV head, then dim); for a fixed (j, kvh) the 256 dims are
 * contiguous. topk_idx is contiguous device I32 [2051, T] (slot, token); slot s of token t holds a
 * key id in [0, n_kv) or -1. positions is contiguous device I32 [T] of absolute query positions.
 * The caller guarantees every non-negative topk id and every position lies in [0, n_kv)
 * and n_kv <= positions[T-1] + 1 (the live prefix), so no device-to-host read is needed.
 *
 * The Op is read-only over q, k, v, topk_idx, and positions; out is completely overwritten. The
 * attention computation is a gather over at most 2051 live keys per (head, token); the numerical
 * result is independent of the gather order up to the Op criterion (stable Softmax over the live
 * set). Workspace holds one small per-block scratch region; `workspace_capacity_bytes` returns the
 * capacity for the inclusive token interval. No persistent state.
 */
void qsa_softmax_attention(const Tensor& q, const Tensor& topk_idx, const Tensor& positions,
                           const Tensor& k, const Tensor& v, std::int32_t n_kv,
                           AttentionHeadGeometry geometry, float scale, WorkspaceArena& workspace,
                           Tensor& out, cudaStream_t stream);

/**
 * Return caller-owned transient capacity for every legal T in the inclusive interval at the
 * registered QSA geometry [256,24,2] and a caller-declared live-key envelope. Invalid profiles or
 * intervals throw; a legal route that needs no global scratch returns zero.
 */
[[nodiscard]] std::size_t qsa_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * QSA q|gate de-interleave (kernel N). The attn_input_proj parent [13312, T] carries its rows in
 * q | k | gate | v order (the converter de-interleaves the source per-head-fused attn_q at write
 * time): the q block is rows [0, 6144) and the gate block rows [6656, 12800), each 24 heads x 256
 * dims with head h / dim d at row h*256 + d (head-major). This Op transposes the q and gate blocks
 * into the dim-major layout qsa_softmax_attention consumes: q, gate are contiguous BF16
 * [256, 24, T] (element (d, h, t) at d*24*T + h*T + t) with
 *
 *   q[d, h, t]    = parent[0     + (h*256 + d)*T + t]
 *   gate[d, h, t] = parent[6656  + (h*256 + d)*T + t]
 *
 * The k/v blocks (rows [6144, 6656) and [12800, 13312)) are not touched; they feed the dense
 * key/value buffer append path. parent is contiguous BF16 [13312, T]; q and gate are contiguous
 * BF16 [256, 24, T] with the same T and are completely overwritten. This is an exact-shape Op
 * (no workspace, no geometry argument); the transformation is bit-exact (a pure data move). The
 * Op is read-only over parent and enqueues on stream without host synchronization.
 */
void qsa_split_qgate(const Tensor& parent, Tensor& q, Tensor& gate, cudaStream_t stream);

/**
 * QSA indexer (K5). The Op owns the whole indexer pipeline over the dense single-stream text
 * side cache (cell j sits at position j): block pooling of the raw cached keys (mean of r=4
 * members, the tail block padded with cell 0), per-block RMSNorm + rope at the block position
 * b*r, per-(head, token) RMSNorm + rope of the query projection at the query position, the
 * 4-head rectified score, the causal / force-in per-cell expansion, and the deterministic top-k.
 *
 * - `k_cache` is contiguous BF16 [128, n_kv]: the raw indexer key for cell j in column j.
 * - `q_raw` is contiguous BF16 [512, T]: the query projection, head h dim k at row 128h + k.
 * - `positions` is contiguous I32 [T]: absolute query positions (positions[t] < n_kv).
 * - `k_norm`, `q_norm` are contiguous BF16 [128] gammas (the Op multiplies by them directly; no
 *   unit offset).  RMS eps is 1e-6; rope uses base 1e7 over the first 64 dims (32 split pairs).
 * - `topk` is contiguous I32 [2051, T]: the selected cell ids in ascending order, padded with -1.
 * - `score` is contiguous FP32 [n_blocks, T] (n_blocks = ceil(n_kv / 4)): the per-block scores,
 *   exposed so a test oracle can verify the selection.
 *
 * The Op needs transient scratch for the pooled/normalized keys [128, n_blocks] and the
 * normalized/rotated queries [512, T]; both are carved from `workspace`.  Call
 * qsa_indexer_workspace_capacity_bytes(n_kv, T) to size it.  The expansion is the dense-text
 * specialization: future cells (j > positions[t]) are -inf, the query's own tail block is
 * force-in at +1e9, and cells in complete earlier blocks carry the block score.  top-k ties
 * break on the lower cell id and the ids are emitted in ascending order.  The Op is read-only
 * over its inputs and enqueues on stream without host synchronization.
 */
void qsa_indexer(const Tensor& k_cache, const Tensor& q_raw, const Tensor& positions,
                 const Tensor& k_norm, const Tensor& q_norm, Tensor& topk, Tensor& score,
                 WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Return the caller-owned transient bytes needed for a qsa_indexer call with the given side-cache
 * length and token count (pooled keys [128, ceil(n_kv/4)] + normalized queries [512, T], both
 * BF16, 256-byte aligned). n_kv and T must be >= 1.
 */
[[nodiscard]] std::size_t qsa_indexer_workspace_capacity_bytes(std::int32_t n_kv, std::int32_t T);

} // namespace ninfer::ops
