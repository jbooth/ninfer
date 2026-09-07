#pragma once

// Real CPU gather routines for the qwen4exp jbq4 target: the token embedding row gather and the
// PLE embedding row gather. The MoE banks need no staging: the decode round (JM4b) points the
// CPU MoE op directly at the page-cache mapping through the HostArtifactView bank spans, and the
// prefill bank ring (JM4c) stages selected expert slots with its own H2D path.

#include <ninfer/targets/qwen4exp/model_view.h>
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen4exp::detail {

// Reads T token embedding rows from the host artifact and H2D's them into `out_bf16`
// ([hidden, T] BF16, hidden-major: dim d of token t at out[d + t*hidden]).
void gather_token_embedding(const HostArtifactView& art, const TokenId* ids, std::int32_t T,
                            std::byte* out_bf16, cudaStream_t stream);

// Gathers the layer-1 PLE residual for B lanes into `out_bf16` ([hidden, B] BF16, hidden-major:
// dim d of lane b at out[d + b*hidden]). `seqs[b]` / `positions[b]` are lane b's sequence and the
// 0-based position of its processed token (the n-gram hash window).
void gather_ple_layer1_batch(const HostArtifactView& art, const TokenId* const* seqs,
                             const std::int32_t* positions, std::int32_t B,
                             std::byte* out_bf16, cudaStream_t stream);

} // namespace ninfer::targets::qwen4exp::detail
