#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::sparse_moe_512x10_detail {

// Runs the full K7 pipeline (router -> topk -> Q4/Q8 gate/up -> Q4/Q8 down ->
// shared Q4/Q8 -> merge) on `stream`. All pointers are device-resident.
// `hidden`/`inter` are fixed at the qwen4exp values (2560 / 640) for the compiled
// kernels. `code` selects the MoE codec: 0 -> Q4G64, 1 -> Q8G64. The base planes
// (codes) and FP16 scale planes are bank-direct (no slot/ring producer).
// Workspace buffers are caller-carved.
void run(const __nv_bfloat16* x, std::int32_t n_experts, std::int32_t top_k, std::int32_t T,
         const float* router, int code, const std::uint8_t* gate_up_codes,
         const std::uint8_t* gate_up_scales, const std::uint8_t* down_codes,
         const std::uint8_t* down_scales, const std::uint8_t* shared_gu_codes,
         const std::uint8_t* shared_gu_scales, const std::uint8_t* shared_down_codes,
         const std::uint8_t* shared_down_scales, __nv_bfloat16* destination, float* scores,
         int* topk_ids, float* topk_w, float* shared_logit, __nv_bfloat16* act, float* y,
         __nv_bfloat16* shared_act, float* shared_y, cudaStream_t stream);

} // namespace ninfer::ops::sparse_moe_512x10_detail
