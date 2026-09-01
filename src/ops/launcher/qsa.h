#pragma once

// ninfer::ops::detail - private launch prototypes for the QSA sparse attention op.

#include "core/tensor.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launches the single registered QSA profile [256,24,2]. The geometry must equal
// {head_dim=256, query_heads=24, kv_heads=2}; the launcher is geometry-exact for v1.
void qsa_softmax_attention_launch(const Tensor& q, const Tensor& topk_idx, const Tensor& positions,
                                  const Tensor& k, const Tensor& v, std::int32_t n_kv,
                                  float scale, Tensor& out, cudaStream_t stream);

// q|gate de-interleave (kernel N): transpose the q and gate blocks of the attn_input_proj parent
// [13312, T] (q | k | gate | v) from head-major to dim-major [256, 24, T]. k/v are not touched.
void qsa_split_qgate_launch(const Tensor& parent, Tensor& q, Tensor& gate, cudaStream_t stream);

// QSA indexer (K5): pool the dense side cache, per-block / per-(head,token) RMSNorm + rope, the
// 4-head rectified score, causal + force-in expansion, and the deterministic top-k.  Writes the
// block scores to score (FP32 [n_blocks, T]) and the top-k cell ids to topk (I32 [2051, T]).
void qsa_indexer_launch(const Tensor& k_cache, const Tensor& q_raw, const Tensor& positions,
                        const Tensor& k_norm, const Tensor& q_norm,
                        Tensor& pooled_out, Tensor& q_out, Tensor& topk, Tensor& score,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
