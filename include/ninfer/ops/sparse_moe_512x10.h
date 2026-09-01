// sparse_moe_512x10 (K7, qwen4exp) Op contract.
//
// A closed, graph-safe Sparse-MoE layer over a resident routed-expert bank, at the qwen4exp
// geometry (512 experts, top-10, hidden 2560, intermediate 640) and its bank codecs (NVFP4
// gate/up, Q6 down, W8 shared expert, FP32 router). It is a sibling of `sparse_moe` (the 35b
// 256/top-8/W8-Q4-Q5-Q6 op), not a profile of it: the routing dtype (FP32) and the NVFP4 routed
// bank make it a distinct closed transformation. The op owns the whole per-token pipeline and is
// launchable from inside a CUDA graph (no host synchronization):
//
//   router logits = router[0..n_experts-1] @ x;  shared_logit = router[n_experts] @ x
//   probs = softmax(logits)                          (FP32, over n_experts)
//   top-k = the `top_k` largest probs, lower-id tie-break (deterministic)
//   w_i = p_i / max(Σ_top_k p, 6.1035e-5)            (norm_w; expert_weights_scale = 0)
//   for each selected expert e:
//       act_e = silu(gate_up_nvfp4[e].gate @ x) * (gate_up_nvfp4[e].up @ x)   [inter]
//       y_e   = down_q6[e] @ act_e                                             [hidden]
//   shared_act = silu(shared_gate_up.gate @ x) * (shared_gate_up.up @ x)
//   shared_y   = shared_down @ shared_act
//   out = Σ_i w_i · y_i + sigmoid(shared_logit) · shared_y
//   destination += out                                  (AddResidual epilogue)
//
// v1 routed GEMMs use the A16 route: the NVFP4/Q6 weights are decoded to FP32 (exact stored-code
// decode) and dotted with the BF16 activation. The NVFP4 `input_divisor` (the paired activation
// scale) is accepted but unused in this route; it is the W4A4 activation quantizer's divisor.

#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Closed MoE geometry. qwen4exp is (512, 10, 2560, 640). The op is generic over this struct so
// the routed GEMMs can be qualified at a reduced expert count on a constrained device; the
// production target instantiates the qwen4exp values. All dimensions must divide the codec group
// sizes (hidden % 32 == 0 for W8/Q6/NVFP4, inter % 32 == 0, inter % 16 == 0 for NVFP4 scale).
struct SparseMoe512x10Geometry {
    std::int32_t n_experts;  // 512 (qwen4exp)
    std::int32_t top_k;      // 10  (qwen4exp)
    std::int32_t hidden;     // 2560 (x dimension; router/gate-up k; down output rows)
    std::int32_t inter;      // 640  (silu_mul dimension; down k)
};

// Resident bank views. Routed banks are row-major over (expert, local-row): expert `e`'s
// gate/up rows are global rows [e * 2*inter, e * 2*inter + 2*inter) (gate [0,inter), up
// [inter,2*inter)); its down rows are global rows [e * hidden, e * hidden + hidden).
//
// routed_gate_up (NVFP4, BlockScaleK16): codes plane [n_experts * 2*inter * hidden / 2] bytes
// (2 E2M1 codes/byte, low nibble = even k); scales plane [n_experts * 2*inter * hidden / 16]
// bytes (one E4M3 scale per 16 k). A weight is decode_e2m1(code) * decode_e4m3(scale).
// routed_down (Q6, row-split K64): codes [n_experts * hidden * (inter/64) * 32], high
// [n_experts * hidden * (inter/64) * 16], scales [n_experts * hidden * (inter/64) * 2].
// shared_gate_up (W8, row-split K32): [2*inter, hidden]; shared_down (W8): [hidden, inter].
struct SparseMoe512x10Weights {
    const float* router;   // FP32 [n_experts + 1, hidden]; row n_experts = shared gate logit
    float input_divisor;   // FP32 scalar (W4A4 activation scale; unused in the A16 route)
    const std::uint8_t* gate_up_codes;
    const std::uint8_t* gate_up_scales;
    const std::uint8_t* down_codes;
    const std::uint8_t* down_high;
    const std::uint8_t* down_scales;
    const std::uint8_t* shared_gate_up_codes;
    const std::uint8_t* shared_gate_up_scales;
    const std::uint8_t* shared_down_codes;
    const std::uint8_t* shared_down_scales;
};

enum class SparseMoe512x10Epilogue { AddResidual };

// x / destination: contiguous BF16 [hidden, T] (destination's incoming value is the residual).
// workspace must hold sparse_moe_512x10_workspace_capacity_bytes(T). All I/O is device-resident;
// the op performs no host synchronization.
void sparse_moe_512x10(const Tensor& x, const SparseMoe512x10Geometry& geo,
                       const SparseMoe512x10Weights& w, SparseMoe512x10Epilogue epilogue,
                       Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t sparse_moe_512x10_workspace_capacity_bytes(
    const SparseMoe512x10Geometry& geo, std::int32_t T);

} // namespace ninfer::ops
