#pragma once

// qwen4exp (Qwen3.8-Flash-Next / jbq4) target geometry.
//
// Dimensions are asserted by the converter preflight (tools/convert/qwen4exp/convert_jbnvfp4.py)
// and cross-checked against the artifact directory (1,390 objects) and the safetensors source of
// truth (/llm/models/Qwen3.8-Flash-Next/). The MoE bank byte facts below are the row-split-k128-v1
// extents measured from the full artifact (per-expert: gu base + scale planes, dn base + scale
// planes) and are the single source for the CPU MoE (JM4b) and the prefill bank ring (JM4c).

#include <cstdint>

namespace ninfer::targets::qwen4exp {

namespace cfg {

// --- text topology ----------------------------------------------------------
inline constexpr std::uint32_t layers       = 48;
inline constexpr std::uint32_t hidden       = 2560;
inline constexpr std::uint32_t hc_dim       = 10240; // 4 streams x 2560
inline constexpr std::uint32_t hc_streams   = 4;
inline constexpr std::uint32_t hc_lr        = 320;   // hyper-connection low-rank width
inline constexpr std::uint32_t vocab        = 248320;

// Layer kind: l % 4 == 3 -> full attention (QSA + indexer); otherwise GDN (linear attention).
inline constexpr std::uint32_t is_full_attn_layer(std::uint32_t l) noexcept {
    return l % 4 == 3;
}
// PLE attaches to the second recurrent layer. The checkpoint config lists it 1-based as
// `ple_layer_ids = [2]`; 0-based this is layer 1 (the converter's PLE_LAYER), which is GDN.
inline constexpr std::uint32_t ple_layer = 1;
inline constexpr bool is_ple_layer(std::uint32_t l) noexcept { return l == ple_layer; }
inline constexpr std::uint32_t full_attn_layers = 12; // l in {3,7,...,47}
inline constexpr std::uint32_t gdn_layers       = 36;

// --- full attention (QSA) ---------------------------------------------------
inline constexpr std::uint32_t qsa_q_heads    = 24;
inline constexpr std::uint32_t qsa_kv_heads   = 2;
inline constexpr std::uint32_t qsa_head_dim   = 256;
inline constexpr std::uint32_t kv_dim         = 512;   // 2 kv heads x 256
inline constexpr std::uint32_t gate_dim       = 6144;  // 24 x 256
inline constexpr std::uint32_t v_dim          = 6144;
// Fused q|k|gate|v projection rows: 6144 + 512 + 6144 + 512 = 13312.
inline constexpr std::uint32_t qkgv_rows      = 13312;
inline constexpr std::uint32_t attn_out_rows  = hidden;
inline constexpr std::uint32_t attn_out_cols  = v_dim;
// Query-sparse indexer: q_proj [512, 2560], k_proj [128, 2560], topk width 2051.
inline constexpr std::uint32_t idx_q_proj     = 512;
inline constexpr std::uint32_t idx_k_dim      = 128;
inline constexpr std::uint32_t idx_topk       = 2051;

// --- GDN (gated delta net, 36 layers) ---------------------------------------
// Recurrent geometry: Hqk = 16, Hv = 48, state dims 128 x 128 (key x value), FP32 state.
//   q|k projection  [4096,  2560] W8   (16 heads x 128, doubled)
//   v|z projection  [12288, 2560] W8   (Hv x 128 = 6144 each)
//   a|b projection  [96,    2560] BF16 (a rows [0,48), b rows [48,96))
//   a_log [48], dt_bias [48] FP32
//   convolution [4, 10240] FP32 (depthwise width 4 over q|k|v|z = 4096 + 6144)
//   norm [128] FP32 (per-channel state norm)
//   output [2560, 6144] W8
inline constexpr std::uint32_t gdn_qk_proj_rows = 4096;
inline constexpr std::uint32_t gdn_vz_proj_rows = 12288;
inline constexpr std::uint32_t gdn_ab_rows      = 96;
inline constexpr std::uint32_t gdn_a_rows       = 48;
inline constexpr std::uint32_t gdn_b_rows       = 48;
inline constexpr std::uint32_t gdn_conv_rows    = 10240;
inline constexpr std::uint32_t gdn_norm_rows    = 128;
inline constexpr std::uint32_t gdn_out_rows     = hidden;
inline constexpr std::uint32_t gdn_out_cols     = v_dim;
inline constexpr std::uint32_t gdn_hqk          = 16;
inline constexpr std::uint32_t gdn_hv           = 48;
inline constexpr std::uint32_t gdn_d            = 128;
inline constexpr std::uint32_t gdn_conv_qk      = 4096;
inline constexpr std::uint32_t gdn_conv_v       = 6144;
inline constexpr std::uint32_t gdn_conv_width   = 4;

// --- MoE (512 routed experts, top-10, shared expert) -------------------------
inline constexpr std::uint32_t moe_experts   = 512;
inline constexpr std::uint32_t moe_ffn       = 640;
inline constexpr std::uint32_t moe_topk      = 10;
inline constexpr std::uint32_t moe_router_rows   = moe_experts + 1; // + shared gate row
inline constexpr std::uint32_t routed_gu_rows    = moe_experts * 2U * moe_ffn;  // 655360
// Routed down is per-expert [hidden, ffn]: 512 x 2560 x 640 (safetensors down_proj
// (512, 2560, 640)). NOT moe_experts x moe_ffn.
inline constexpr std::uint32_t routed_dn_rows    = moe_experts * hidden;        // 1310720
inline constexpr std::uint32_t shared_gu_rows    = 2U * moe_ffn;                // 1280
inline constexpr std::uint32_t shared_dn_rows    = hidden;                      // 2560

// D14 codec map: layers {0, 1, 47} and the MTP layer use Q8G64_F16S; the other 45 text layers
// use Q4G64_F16S.
[[nodiscard]] constexpr bool moe_layer_is_q8(std::uint32_t layer) noexcept {
    return layer == 0 || layer == 1 || layer == layers - 1;
}
inline constexpr bool mtp_layer_is_q8 = true;

// Row-split-k128-v1 bank extents (measured from the full artifact; one expert's gu = base
// [2*ffn, hidden/2] + scale [2*ffn, hidden/64] FP16, dn = base [hidden, ffn/2] + scale
// [hidden, ffn/64]).
inline constexpr std::uint64_t expert_gu_base_bytes_q4  = 2U * moe_ffn * (hidden / 2U);  // 1,638,400
inline constexpr std::uint64_t expert_gu_scale_bytes_q4 = 2U * moe_ffn * (hidden / 64U) * 2U; // 102,400
inline constexpr std::uint64_t expert_dn_base_bytes_q4  = hidden * (moe_ffn / 2U);        // 819,200
inline constexpr std::uint64_t expert_dn_scale_bytes_q4 = hidden * (moe_ffn / 64U) * 2U;  // 51,200
inline constexpr std::uint64_t expert_gu_bytes_q4  = expert_gu_base_bytes_q4 + expert_gu_scale_bytes_q4; // 1,740,800
inline constexpr std::uint64_t expert_dn_bytes_q4  = expert_dn_base_bytes_q4 + expert_dn_scale_bytes_q4; // 870,400
inline constexpr std::uint64_t expert_gu_bytes_q8  = 2U * moe_ffn * hidden + expert_gu_scale_bytes_q4;  // 3,379,200
inline constexpr std::uint64_t expert_dn_bytes_q8  = hidden * moe_ffn + expert_dn_scale_bytes_q4;        // 1,689,600

// Per-layer routed bank extents (all experts flattened; base plane then scale plane).
inline constexpr std::uint64_t routed_gu_bytes_q4 = expert_gu_bytes_q4 * moe_experts; // 891,289,600
inline constexpr std::uint64_t routed_gu_bytes_q8 = expert_gu_bytes_q8 * moe_experts; // 1,730,150,400
inline constexpr std::uint64_t routed_dn_bytes_q4 = expert_dn_bytes_q4 * moe_experts; // 445,644,800
inline constexpr std::uint64_t routed_dn_bytes_q8 = expert_dn_bytes_q8 * moe_experts; // 865,075,200
inline constexpr std::uint64_t router_bytes       = moe_router_rows * hidden * 4U;    // 5,253,120

// Per-(layer, token) touched routed bytes at top-10 (both codecs).
inline constexpr std::uint64_t routed_touched_per_token_q4 =
    moe_topk * (expert_gu_bytes_q4 + expert_dn_bytes_q4); // 26,112,000
inline constexpr std::uint64_t routed_touched_per_token_q8 =
    moe_topk * (expert_gu_bytes_q8 + expert_dn_bytes_q8); // 50,688,000

// Expert scratch slot for the prefill bank ring (JM4c): one routed expert's gu + dn
// (base + scale planes), sized for the larger codec (Q8).
inline constexpr std::uint64_t expert_scratch_bytes =
    (moe_layer_is_q8(0) ? expert_gu_bytes_q8 + expert_dn_bytes_q8 : expert_gu_bytes_q4 + expert_dn_bytes_q4);
inline constexpr std::uint32_t expert_scratch_slots = 192;
inline constexpr std::uint64_t expert_scratch_bank_bytes = expert_scratch_slots * expert_scratch_bytes;

// --- MTP (speculative draft layer) ------------------------------------------
// Single draft layer: fc_embedding / fc_hidden W8 [2560, 2560], fused qkgv W8 [13312, 2560]
// (same QSA geometry), attention output W8 [2560, 6144], indexer W8 (q [512,2560], k [128,2560]),
// two HC mix blocks (W8 up/down [10240,320]/[320,10240], inject [4,10240]), and the MoE with
// Q8 banks. KV per draft token: k|v = 2 x 512 x 2B = 2048 B; side cache: indexer k = 256 B.
inline constexpr std::uint32_t mtp_qkgv_rows      = qkgv_rows;
inline constexpr std::uint32_t mtp_attn_out_rows  = hidden;
inline constexpr std::uint32_t mtp_attn_out_cols  = v_dim;
inline constexpr std::uint32_t mtp_fc_rows        = hidden;
inline constexpr std::uint32_t mtp_idx_q_rows     = idx_q_proj;
inline constexpr std::uint32_t mtp_idx_k_rows     = idx_k_dim;
inline constexpr std::uint32_t mtp_kv_bytes_per_token  = 2U * kv_dim * 2U;          // 2048
inline constexpr std::uint32_t mtp_side_bytes_per_token = idx_k_dim * 2U;           // 256

// --- PLE (personalization-layer embedding) -----------------------------------
inline constexpr std::uint64_t ple_rows        = 320001536;
inline constexpr std::uint32_t ple_dim         = 160;
inline constexpr std::uint32_t ple_heads       = 32;
inline constexpr std::uint32_t ple_multipliers = 6;
inline constexpr std::uint32_t ple_heads_per_ngram = 8;
inline constexpr std::uint32_t ple_ngram_size  = 3;
inline constexpr std::int64_t  ple_eos         = 248044;
// PLE dense projections (shared across PLE layers): key [hc_dim, hidden], value [hidden, hidden],
// convolution [4, hc_dim], three FP32 norms [hc_dim].
inline constexpr std::uint32_t ple_key_rows    = hc_dim;
inline constexpr std::uint32_t ple_value_rows  = hidden;
inline constexpr std::uint32_t ple_conv_rows   = hc_dim;

// --- Vision (validate-only; bound for completeness) --------------------------
inline constexpr std::uint32_t vision_blocks  = 27;
inline constexpr std::uint32_t vision_hidden  = 1152;
inline constexpr std::uint32_t vision_mlp_intermediate = 4304; // fc1/fc2 width (per artifact)
inline constexpr std::uint32_t vision_patch_input = 1536;
inline constexpr std::uint32_t vision_pos_rows = 2304;
inline constexpr std::uint32_t vision_merger_intermediate = 4U * vision_hidden;   // 4608

// --- Capacity accounting ------------------------------------------------------
// Per-token main KV bytes: full_attn_layers x (k+v) x 2B = 12 x 1024 x 2 = 24576.
inline constexpr std::uint32_t main_kv_bytes_per_token =
    full_attn_layers * (kv_dim + kv_dim) * 2U;
// Per-token side cache bytes: full_attn_layers x idx_k_dim x 2B = 12 x 128 x 2 = 3072.
inline constexpr std::uint32_t side_cache_bytes_per_token =
    full_attn_layers * idx_k_dim * 2U;
// Per-sequence GDN recurrent state (FP32): 36 x 48 x 128 x 128 x 4 = 113,246,208.
inline constexpr std::uint64_t gdn_state_bytes_per_sequence =
    std::uint64_t{gdn_layers} * gdn_hv * gdn_d * gdn_d * 4U;
// Per-sequence GDN conv state (BF16): 36 x (4096 + 6144) x (width-1) x 2 = 2,228,224.
inline constexpr std::uint64_t gdn_conv_state_bytes_per_sequence =
    std::uint64_t{gdn_layers} * (gdn_conv_qk + gdn_conv_v) * (gdn_conv_width - 1U) * 2U;
// Per-sequence PLE dilated-conv history (BF16): 9 taps x hc_dim x 2B = 184,320.
inline constexpr std::uint64_t ple_conv_history_bytes_per_sequence = std::uint64_t{9} * hc_dim * 2U;

// Prefill chunk width for the eager dataflow.
inline constexpr std::uint32_t prefill_chunk_tokens = 64;

} // namespace cfg
} // namespace ninfer::targets::qwen4exp
