#pragma once

// Real CPU gather / streaming routines for the qwen4exp skeleton. These are the only non-sampled
// execution paths that do real work in phase 2: the token embedding gather, the layer-1 PLE
// residual gather, and the routed-MoE expert slot streaming (D2H router + pread + H2D).

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

// Streams the `topk` selected routed experts for `layer` from the host artifact into the device
// scratch banks (`gu_scratch`, `dn_scratch`), each `topk` slots of the per-expert slot size.
void stream_experts(const HostArtifactView& art, std::int32_t layer, const TokenId* router_row_dst,
                    const std::uint8_t* selected_experts, std::int32_t topk, std::byte* gu_scratch,
                    std::byte* dn_scratch, cudaStream_t stream);

} // namespace ninfer::targets::qwen4exp::detail
