// include/ninfer/ops/sparse_moe_512x10_geometry.h
//
// CUDA-free shared geometry for the Qwen4Exp 512-expert top-10 routed MoE (jbq4). The GPU
// prefill route (sparse_moe_512x10) and the CPU decode route (sparse_moe_512x10_cpu) share
// this model shape so they agree on the bank dimensions.
#pragma once

#include <cstdint>

namespace ninfer::ops {

// Per-layer MoE codec tag. Only the base-plane encoding differs; the FP16 scale plane,
// group size (64), and row-split geometry are shared.
enum class MoeCode : std::uint8_t {
    Q4G64 = 0,
    Q8G64 = 1,
};

struct SparseMoe512x10Geometry {
    std::int32_t n_experts = 512;  // routed experts (reduced in tests)
    std::int32_t top_k     = 10;   // experts selected per token (<= 16)
    std::int32_t hidden    = 2560; // H
    std::int32_t inter     = 640;  // I
};

}  // namespace ninfer::ops
