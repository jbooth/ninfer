// include/ninfer/ops/sparse_moe_512x10_cpu.h
//
// Host-only (decode) closed Op for the Qwen4Exp 512-expert top-10 routed MoE (jbq4).
// This is the CPU twin of `sparse_moe_512x10` (the GPU prefill route, K7). It runs the
// exact same closed math (f32 router -> f32 softmax -> top-10 lower-id tie-break ->
// w_i = p_i / max(sum, 2^-14) -> per-expert Q4/Q8-decoded GEMMs -> residual merge) over the
// row-split-k128-v1, group-64, FP16-scale MoE banks, using an integer (W4A8/W8A8) GEMM:
// the input activation is quantized to int8 once per token, and each expert's swiglu
// intermediate is quantized to int8 before its down GEMM. Both the Q4 (packed nibble) and
// Q8 (raw i8) banks decode to the same signed code, so one integer GEMM core serves both.
//
// The Op is host-resident and multi-threaded. It is bit-deterministic for a fixed
// (input, bank): the per-(row, token) router, the per-expert GEMM (k sequential), and the
// merge (experts in ascending id, shared last) each use a fixed association order, so no
// reduction order depends on thread partitioning.
#pragma once

#include "ninfer/ops/sparse_moe_512x10_geometry.h"

#include <cstdint>

namespace ninfer::ops {

struct SparseMoe512x10CpuWeights {
    const float* router = nullptr;  // FP32 [n_experts + 1, hidden]; row n_experts = shared gate

    // Routed bank (bank-direct; the page-cache bank reader is a caller concern).
    MoeCode             routed_gate_up_codec = MoeCode::Q4G64;
    const std::uint8_t* routed_gate_up_base = nullptr;     // [n_experts * 2 * inter, hidden]
    const std::uint16_t* routed_gate_up_scales = nullptr;   // FP16 [n_experts * 2 * inter, hidden/64]
    MoeCode             routed_down_codec = MoeCode::Q4G64;
    const std::uint8_t* routed_down_base = nullptr;        // [n_experts * hidden, inter]
    const std::uint16_t* routed_down_scales = nullptr;      // FP16 [n_experts * hidden, inter/64]

    // Shared expert (always runs; gated by sigmoid of the shared router row).
    MoeCode             shared_gate_up_codec = MoeCode::Q4G64;
    const std::uint8_t* shared_gate_up_base = nullptr;     // [2 * inter, hidden]
    const std::uint16_t* shared_gate_up_scales = nullptr;   // FP16 [2 * inter, hidden/64]
    MoeCode             shared_down_codec = MoeCode::Q4G64;
    const std::uint8_t* shared_down_base = nullptr;        // [hidden, inter]
    const std::uint16_t* shared_down_scales = nullptr;      // FP16 [hidden, inter/64]
};

// destination[i * T + t] += (sum over selected experts of w[rank] * down[expert][i][t])
//                            + sigmoid(shared_logit[t]) * shared_down[i][t].
// x and destination are BF16 (int16 bits) [hidden, T], host-resident, column-major over t
// (column t is x[hidden * t ..]). The router uses the exact BF16 activation (so it agrees
// with the GPU router); the integer GEMMs use the int8-quantized activation.
void sparse_moe_512x10_cpu(const std::int16_t* x, std::int32_t T,
                           const SparseMoe512x10Geometry& geo,
                           const SparseMoe512x10CpuWeights& w, std::int16_t* destination,
                           int thread_count = 0);  // 0 -> hardware concurrency

}  // namespace ninfer::ops
