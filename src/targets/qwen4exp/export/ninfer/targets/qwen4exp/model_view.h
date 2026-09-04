#pragma once

// qwen4exp (jbq4) model view: the host page-cache view over the streaming artifact objects (token
// / PLE embedding rows, routed + shared MoE banks, router gates), the owned frontend resources,
// and the geometry-derived capacity accessors.
//
// The MoE banks are accessed zero-copy through a MAP_PRIVATE read mapping of the artifact:
// the OS page cache serves repeated reads, so a decode step touches ~1.38 GB of pages without
// any staging copy or explicit I/O (see q4plan.md §6.3). The small per-token rows (token
// embedding, PLE table) use aligned pread on the owned fd.
//
// Execution-phase note: the Program runs the 48-layer decode dataflow with the real CPU MoE
// op (`ninfer::ops::sparse_moe_512x10_cpu`) over these bank views (JM4b).

#include "ninfer/types.h"
#include <ninfer/targets/qwen4exp/runtime.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/ops/sparse_moe_512x10_geometry.h>

#include "core/tensor.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <span>

namespace ninfer::targets::qwen4exp {
namespace detail {
using ninfer::ops::MoeCode;
// Self-contained geometry subset for the model view (exported; must not collide with the impl
// ``qwen4exp::cfg``). HostArtifactView / ModelView are in the same namespace and use these directly.
inline constexpr std::uint32_t layers      = 48;
inline constexpr std::uint32_t hidden      = 2560;
inline constexpr std::uint32_t hc_dim      = 10240;
inline constexpr std::uint32_t hc_lr       = 320;
inline constexpr std::uint32_t gate_dim    = 6144;
inline constexpr std::uint32_t v_dim       = 6144;
inline constexpr std::uint32_t kv_dim      = 512;
inline constexpr std::uint32_t idx_k_dim   = 128;
inline constexpr std::uint32_t idx_topk    = 2051;
inline constexpr std::uint32_t gdn_layers  = 36;
inline constexpr std::uint32_t full_attn   = 12;
inline constexpr std::uint32_t gdn_hv      = 48;
inline constexpr std::uint32_t gdn_d       = 128;
inline constexpr std::uint32_t gdn_conv_qk = 4096;
inline constexpr std::uint32_t gdn_conv_v  = 6144;
inline constexpr std::uint32_t moe_experts = 512;
inline constexpr std::uint32_t moe_ffn     = 640;
inline constexpr std::uint32_t moe_topk    = 10;
inline constexpr std::uint64_t ple_rows    = 320001536;
inline constexpr std::uint32_t ple_dim     = 160;
inline constexpr std::int64_t  ple_eos     = 248044;
inline constexpr std::uint32_t vocab       = 248320;

// A file-offset plane span within one artifact object (or object region).
struct BankPlane {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

// One MoE expert bank pair (gate_up + down) as base/scale planes of the row-split-k128-v1
// object, for one codec. Offsets are absolute file offsets.
struct MoEBank {
    MoeCode code = MoeCode::Q4G64;
    BankPlane gate_up_base;
    BankPlane gate_up_scales;
    BankPlane down_base;
    BankPlane down_scales;
    [[nodiscard]] bool present() const noexcept { return gate_up_base.bytes != 0; }
};

// Host page-cache view over the streaming (validate-only) artifact objects.
class HostArtifactView {
public:
    HostArtifactView() = default;
    ~HostArtifactView();

    HostArtifactView(const HostArtifactView&)            = delete;
    HostArtifactView& operator=(const HostArtifactView&) = delete;
    HostArtifactView(HostArtifactView&&) noexcept;
    HostArtifactView& operator=(HostArtifactView&&) noexcept;

    [[nodiscard]] bool valid() const noexcept { return fd >= 0; }

    // Reads exactly `bytes` from absolute file offset `off` into `dst` (aligned pread; retries
    // EINTR; throws on a short read or I/O error).
    void read_at(std::uint64_t off, void* dst, std::size_t bytes) const;

    // Zero-copy pointer into the page-cache mapping for absolute file offset `off`. The mapping
    // is MAP_PRIVATE read-only over the whole artifact; pages fault in on first touch and the OS
    // manages eviction. Valid for the lifetime of this view.
    [[nodiscard]] const std::byte* map_ptr(std::uint64_t off) const noexcept;

    std::uint64_t file_bytes = 0;

    // token embedding [vocab, hidden] BF16
    std::uint64_t token_emb_off       = 0;
    std::uint32_t token_emb_row_bytes = 0;
    // PLE table [ple_rows, ple_dim] BF16
    std::uint64_t ple_off       = 0;
    std::uint32_t ple_row_bytes = 0;

    // PLE descriptors. Each stored I32 u64 pair is little-endian (lo, hi). The artifact carries
    // 6 i32 n-gram multipliers (3 u64) and 32 i32 head offsets / 32 i32 head vocab sizes (16 u64).
    std::array<std::uint64_t, 3>  ple_multipliers{};
    std::array<std::uint64_t, 16> ple_head_offsets{};
    std::array<std::uint64_t, 16> ple_head_vocab_sizes{};

    // Per-layer object presence (full artifact: all true; smoke carries a subset). Layer kind is
    // fixed by the geometry (l % 4 == 3 -> full attention); presence covers the whole 21-object
    // template together with the MoE objects.
    std::array<bool, layers> layer_present{};
    bool                    mtp_present = false;
    bool                    vision_present = false;

    // Routed expert banks, per layer (page-cache; base plane then scale plane inside each object).
    std::array<MoEBank, layers> routed{};
    // MTP routed bank (always Q8 in jbq4).
    MoEBank mtp_routed{};

    // MoE side (router gate + shared expert) per layer, and the MTP side. The router is also
    // device-materialized (prefill GPU route); the CPU decode route reads these through map_ptr.
    struct MoeSide {
        BankPlane router;          // FP32 [513, hidden]
        MoEBank   shared;          // shared expert gate_up / down (codec matches the layer)
    };
    std::array<MoeSide, layers> moe_side{};
    MoeSide                     mtp_moe_side{};

    // Owned fd (O_RDONLY) for pread streaming reads. Package-internal.
    int fd = -1;
    // Owned MAP_PRIVATE read mapping of the whole artifact (zero-copy bank access).
    void*         map_base    = nullptr;
    std::uint64_t map_bytes   = 0;
    std::uint64_t payload_off = 0; // file offset of the first payload byte (reader metadata size)
};

// GPU-resident dense/quantized weight views for the 48 text layers, the root output head,
// and the shared PLE dense block. Filled from the materialized artifact at load time
// (JM4b load-side). The MoE routed banks, shared banks, PLE table, and vision stay page-cache
// (HostArtifactView); only the router gate is device-resident (prefill GPU route). MTP device
// weights are deferred to JM5.
struct DeviceWeights {
    struct GdnBlock {
        Weight query_key;       // W8 [4096, 2560]
        Weight value_z;         // W8 [12288, 2560]
        Tensor a_b_projection;  // BF16 [96, 2560]
        Tensor a_log;           // FP32 [48]
        Tensor dt_bias;         // FP32 [48]
        Tensor convolution;     // FP32 [4, 10240]
        Tensor norm;            // FP32 [128]
        Weight output;          // W8 [2560, 6144]
    };
    struct QsaBlock {
        Weight query_key_gate_value; // W8 [13312, 2560]
        Weight output;               // W8 [2560, 6144]
        Tensor query_norm;           // FP32 [256]
        Tensor key_norm;             // FP32 [256]
        Tensor indexer_query_proj;   // BF16 [512, 2560]
        Tensor indexer_key_proj;     // BF16 [128, 2560]
        Tensor indexer_query_norm;   // FP32 [128]
        Tensor indexer_key_norm;     // FP32 [128]
    };
    // One HC unit (attn or ffn): the low-rank mixer (down res_hc->320, up 320->res_hc), the
    // grouped RMSNorm gamma, and the 4-stream inject residual.
    struct Hc {
        Tensor up;     // BF16 [10240, 320]
        Tensor down;   // BF16 [320, 10240]
        Tensor norm;   // FP32 [10240]
        Tensor inject; // BF16 [4, 10240]
    };
    struct Layer {
        GdnBlock gdn;
        QsaBlock qsa;
        Hc       attn_hc;
        Hc       ffn_hc;
        Tensor   router; // FP32 [513, 2560]
    };
    std::array<Layer, layers> per_layer{};

    // Root.
    Tensor output_head; // BF16 [248320, 2560]
    Hc     output_hc;   // up [10240, 320], down [320, 10240], norm [10240], inject [4, 10240]

    // Shared PLE dense block (injected at layer 1).
    Tensor ple_key;       // BF16 [10240, 2560]
    Tensor ple_value;     // BF16 [2560, 2560]
    Tensor ple_conv;      // BF16 [4, 10240]
    Tensor ple_norm_query; // FP32 [10240]
    Tensor ple_norm_key;   // FP32 [10240]
    Tensor ple_norm_conv;  // FP32 [10240]
};

struct ModelView {
    HostArtifactView host;
    DeviceWeights    weights{};
    qwen3_6::FrontendResources frontend;

    // Per-sequence context-independent state byte budget (capacity accounting).
    [[nodiscard]] constexpr std::uint64_t gdn_state_bytes_per_sequence() const noexcept {
        // gdn_layers x hv x d x d x 4B (FP32 recurrent state).
        return std::uint64_t{gdn_layers} * gdn_hv * gdn_d * gdn_d * 4U;
    }
    [[nodiscard]] constexpr std::uint64_t gdn_conv_state_bytes_per_sequence() const noexcept {
        // gdn_layers x (qk 4096 + v 6144) x width-3 window x 2B (BF16).
        return std::uint64_t{gdn_layers} * (gdn_conv_qk + gdn_conv_v) * 3U * 2U;
    }
    [[nodiscard]] constexpr std::uint64_t ple_conv_history_bytes_per_sequence() const noexcept {
        // PLE dilated conv history: 9 taps x hc_dim x 2B (BF16).
        return std::uint64_t{9} * hc_dim * 2U;
    }
    [[nodiscard]] constexpr std::uint32_t main_kv_bytes_per_token() const noexcept {
        // full_attn x (k+v) x 2B = 12 x 1024 x 2.
        return full_attn * (kv_dim + kv_dim) * 2U;
    }
    [[nodiscard]] constexpr std::uint32_t side_cache_bytes_per_token() const noexcept {
        // full_attn x idx_k_dim x 2B = 12 x 128 x 2.
        return full_attn * idx_k_dim * 2U;
    }
    [[nodiscard]] constexpr std::uint32_t mtp_kv_bytes_per_token() const noexcept {
        // MTP k|v = 2 x kv_dim x 2B = 2048.
        return 2U * kv_dim * 2U;
    }
    [[nodiscard]] constexpr std::uint32_t mtp_side_cache_bytes_per_token() const noexcept {
        // MTP indexer k = idx_k_dim x 2B = 256.
        return idx_k_dim * 2U;
    }
};

} // namespace detail

} // namespace ninfer::targets::qwen4exp
