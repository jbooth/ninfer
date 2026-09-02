#pragma once
//
// include/ninfer/ops/sparse_moe_512x10.h — closed Op contract for the Qwen4Exp
// (Qwen3.8-Flash-Next) 512-expert top-10 routed MoE (jbq4), GPU prefill route.
//
// One closed unit, called once per MoE layer:
//   scores   = router [n_experts + 1, hidden] @ x [hidden, T]   (FP32 weights)
//   probs    = softmax(scores[0..n_experts-1])
//   topk     = top-`top_k`(probs)                    (lower expert id wins ties)
//   topk_w   = probs[topk] / max(sum(probs[topk]), 2^-14)       (norm_w)
//   act_e    = swiglu(routed_gate_up[e] @ x)        per selected expert e
//   y_e      = routed_down[e] @ act_e
//   shared   = sigmoid(scores[n_experts]) * (shared_down @ swiglu(shared_gate_up @ x))
//   out      = residual + sum_e topk_w_e * y_e + shared
//
// The routed and shared MoE weights are row-split-k128-v1, group size 64, no high
// plane, FP16 scale plane. Two codecs are supported, selected per layer by
// `MoeCode`:
//   Q4G64: base plane = packed nibbles (2 codes/byte, decode (nib^8)-8), 32 B/group.
//   Q8G64: base plane = raw i8 (1 code/byte), 64 B/group.
// The FP16 scale plane is identical (one scale per 64-wide group). The NVFP4
// `input_divisor` of the retired design is gone.
//
// The GEMM front-ends are decode-oriented (a warp per output element, lanes
// striding the reduction over K). This is the kernel-level, bank-direct contract;
// the slot/ring producer, CUDA-graph capture, and target wiring are the dataflow
// scope. `destination` is pre-seeded with the residual; the op adds the MoE
// contribution (AddResidual).
//
// Mathematical reference: llama.cpp src/models/qwen4exp.cpp::build_layer_ffn +
// llama-graph.cpp::build_moe_ffn (softmax gating, norm_w, shared-expert sigmoid
// gate). The lower-id top-k tie-break is the house rule.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe_512x10_geometry.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SparseMoe512x10Weights {
    const float* router = nullptr;  // FP32 [n_experts + 1, hidden] (row n_experts = shared gate)

    // Routed bank (bank-direct; the slot/ring producer is a caller concern).
    const void* routed_gate_up_codes = nullptr;  // [n_experts * 2 * inter, hidden] base plane
    const void* routed_gate_up_scales = nullptr; // scale plane
    const void* routed_down_codes     = nullptr;  // [n_experts * hidden, inter] base plane
    const void* routed_down_scales    = nullptr;  // scale plane
    // Shared expert (always active, its own sigmoid gate).
    const void* shared_gate_up_codes  = nullptr;  // [2 * inter, hidden] base plane
    const void* shared_gate_up_scales = nullptr;  // scale plane
    const void* shared_down_codes     = nullptr;  // [hidden, inter] base plane
    const void* shared_down_scales    = nullptr;  // scale plane

    MoeCode code = MoeCode::Q4G64;  // applied to all four MoE banks of this layer
};

enum class SparseMoe512x10Epilogue : std::uint8_t {
    AddResidual,
};

// Exact private-workspace byte capacity for one call.
[[nodiscard]] std::size_t sparse_moe_512x10_workspace_capacity_bytes(
    const SparseMoe512x10Geometry& geo, std::int32_t T);

// x [hidden, T] BF16 (row-major, column t is x[hidden * t ..]). destination
// [hidden, T] BF16, pre-seeded with the residual (the op adds in place). All
// intermediate buffers are private to the Op and drawn from `workspace` on
// `stream`. The MoE weights must match `geo` (hidden/inter fixed at 2560/640).
void sparse_moe_512x10(const Tensor& x, const SparseMoe512x10Geometry& geo,
                       const SparseMoe512x10Weights& w, SparseMoe512x10Epilogue epilogue,
                       Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

}  // namespace ninfer::ops
