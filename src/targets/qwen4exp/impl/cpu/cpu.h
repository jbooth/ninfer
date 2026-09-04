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
// ([T, hidden] BF16, row-major).
void gather_token_embedding(const HostArtifactView& art, const TokenId* ids, std::int32_t T,
                            std::byte* out_bf16, cudaStream_t stream);

// Builds the layer-1 PLE residual row for position `i` (16 heads x 160 dim -> [16*160] BF16) into
// `out_bf16` (16 rows of ple_dim). `ids` is the token sequence and `T` its length.
void gather_ple_layer1(const HostArtifactView& art, const TokenId* ids, std::int32_t i,
                       std::int32_t T, std::byte* out_bf16, cudaStream_t stream);

} // namespace ninfer::targets::qwen4exp::detail
