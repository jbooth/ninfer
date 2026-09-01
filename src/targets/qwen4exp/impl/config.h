#pragma once

#include <cstdint>

namespace ninfer::targets::qwen4exp {

// Geometry asserted by the converter preflight (tools/convert/qwen4exp/convert_jbnvfp4.py).
// These are the authoritative dimensions for the qwen3.8-flash-next / jbnvfp4 artifact.
namespace cfg {

inline constexpr std::uint32_t layers       = 48;
inline constexpr std::uint32_t hidden       = 2560;
inline constexpr std::uint32_t hc_dim       = 10240; // 4 streams of 2560
inline constexpr std::uint32_t hc_streams   = 4;
inline constexpr std::uint32_t hc_lr        = 320;   // HC low-rank width

inline constexpr std::uint32_t heads        = 24;
inline constexpr std::uint32_t head_dim     = 256;
inline constexpr std::uint32_t kv_heads     = 2;
inline constexpr std::uint32_t kv_dim       = 512;   // 2 kv heads x 256
inline constexpr std::uint32_t gate_dim     = 6144;
inline constexpr std::uint32_t v_dim        = 6144;
inline constexpr std::uint32_t qk_dim       = 4096;  // GDN fused q|k
inline constexpr std::uint32_t value_z_dim  = 12288; // GDN fused v|z

inline constexpr std::uint32_t idx_k_dim    = 128;   // QSA indexer k
inline constexpr std::uint32_t idx_q_proj   = 512;   // QSA indexer q_proj rows
inline constexpr std::uint32_t idx_topk     = 2051;  // QSA topk_idx width

// Full-attn (QSA) layer index pattern: l % 4 == 3. GDN otherwise.
inline constexpr std::uint32_t full_attn_layers = 12; // l in {3,7,...,47}
inline constexpr std::uint32_t gdn_layers         = 36; // l % 4 != 3

// GDN recurrent geometry (state dim fixed 128).
inline constexpr std::uint32_t gdn_hqk = 16;
inline constexpr std::uint32_t gdn_hv  = 48;
inline constexpr std::uint32_t gdn_d   = 128;
// GDN a_b_projection fused rows: alpha [0,48), beta [48,96).
inline constexpr std::uint32_t gdn_ab_rows = 96;
// GDN conv width.
inline constexpr std::uint32_t gdn_conv_width = 4;
// GDN conv channel extents: qk conv C=4096, v conv C=6144.
inline constexpr std::uint32_t gdn_conv_qk = 4096;
inline constexpr std::uint32_t gdn_conv_v  = 6144;

// MoE geometry.
inline constexpr std::uint32_t moe_experts   = 512;
inline constexpr std::uint32_t moe_ffn       = 640;
inline constexpr std::uint32_t moe_topk      = 10;
inline constexpr std::uint32_t moe_gate_up   = 1280;  // 2 x 640 (gate|up) per expert
inline constexpr std::uint32_t moe_router_rows = 513; // 512 experts + 1 shared gate
inline constexpr std::uint32_t moe_shared_gate_up = 1280;
inline constexpr std::uint32_t moe_shared_down      = 2560;
// Routed-bank per-layer extents (flattened over experts):
//   routed_gate_up : [moe_experts*moe_gate_up, hidden] = [655360, 2560] NVFP4
//   routed_down    : [moe_experts*moe_ffn,     hidden] = [1310720, 640]  Q6
inline constexpr std::uint32_t routed_gate_up_rows = moe_experts * moe_gate_up;
inline constexpr std::uint32_t routed_down_rows    = moe_experts * moe_ffn;

// PLE geometry.
inline constexpr std::uint64_t ple_rows  = 320001536;
inline constexpr std::uint32_t ple_dim   = 160;
inline constexpr std::uint32_t ple_heads = 32;
inline constexpr std::uint32_t ple_multipliers = 6;
inline constexpr std::uint32_t ple_heads_per_ngram = 8;
inline constexpr std::uint32_t ple_ngram_size     = 3;
inline constexpr std::int64_t  ple_eos            = 248044;

// Full-attn fused QKQV projection row count = q 6144 + k 512 + gate 6144 + v 512 = 13312.
inline constexpr std::uint32_t attn_qkv_rows = gate_dim + kv_dim + gate_dim + kv_dim;

// Attention output projection (post-mixer) [hidden, v_dim] W8.
inline constexpr std::uint32_t attn_out_rows = hidden;
inline constexpr std::uint32_t attn_out_cols = v_dim;

// Output head.
inline constexpr std::uint32_t vocab = 248320;

// Per-token main KV bytes: full_attn_layers x (k+v) x 2B = 12 x 1024 x 2 = 24576.
inline constexpr std::uint32_t main_kv_bytes_per_token =
    full_attn_layers * (kv_dim + kv_dim) * 2U;
// Per-token QSA side-cache bytes: full_attn_layers x idx_k_dim x 2B = 12 x 128 x 2 = 3072.
inline constexpr std::uint32_t side_cache_bytes_per_token =
    full_attn_layers * idx_k_dim * 2U;

// Per-sequence context-independent GDN recurrent state bytes (FP32):
// gdn_layers x hv x d x d x 4B = 36 x 48 x 128 x 128 x 4 = 113,246,208.
inline constexpr std::uint64_t gdn_state_bytes_per_sequence =
    std::uint64_t{gdn_layers} * gdn_hv * gdn_d * gdn_d * 4U;
// Per-sequence GDN conv state bytes (BF16):
// gdn_layers x (qk 4096 + v 6144) x width-3 window x 2B = 36 x 10240 x 3 x 2 = 2,228,224.
inline constexpr std::uint64_t gdn_conv_state_bytes_per_sequence =
    std::uint64_t{gdn_layers} * (gdn_conv_qk + gdn_conv_v) * (gdn_conv_width - 1U) * 2U;

// PLE dilated conv history per sequence (BF16): width 4 x dilation 3 -> 9 taps x hc_dim x 2B.
inline constexpr std::uint64_t ple_conv_history_bytes_per_sequence = std::uint64_t{9} * hc_dim * 2U;

// Expert scratch bank (v1 streaming invariant). One slot holds one routed expert's
// gate (NVFP4 921,600) + up (NVFP4 921,600) + down (Q6 1,280,000) = 3,123,200 B.
inline constexpr std::uint64_t expert_scratch_bytes = 3123200U;
inline constexpr std::uint32_t expert_scratch_slots = 192;
inline constexpr std::uint64_t expert_scratch_bank_bytes = expert_scratch_slots * expert_scratch_bytes;

// Prefill chunk width for the eager dataflow.
inline constexpr std::uint32_t prefill_chunk_tokens = 64;

inline constexpr std::uint32_t is_full_attn_layer(std::uint32_t layer) { return layer % 4 == 3; }

} // namespace cfg
} // namespace ninfer::targets::qwen4exp
