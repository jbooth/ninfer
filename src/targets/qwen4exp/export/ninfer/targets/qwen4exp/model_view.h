#pragma once

// qwen4exp model view: the host page-cache streaming view (token/PLE embedding + routed expert
// banks), the owned frontend resources, and the geometry-derived capacity accessors.
//
// Phase-2 note: the GEMM-class ops and the stateful kernels are target-owned constant-fill stubs
// (see phase2.md §16). The device weights are still bound and materialized into the arena (binder
// validation + real residency for the "fits 32 GB" gate) but are not individually addressed, so no
// per-weight device views are kept here. Only the small PLE descriptor vectors (I32) are lifted to
// host at bind time for the real CPU gather routines.

#include "ninfer/types.h"
#include <ninfer/targets/qwen4exp/runtime.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include <array>
#include <cstdint>
#include <cstddef>
#include <span>

namespace ninfer::targets::qwen4exp {
namespace detail {
// Self-contained geometry subset for the model view (exported; must not collide with the impl
// ``qwen4exp::cfg``). HostArtifactView / ModelView are in the same namespace and use these directly.
inline constexpr std::uint32_t layers      = 48;
inline constexpr std::uint32_t hidden      = 2560;
inline constexpr std::uint32_t hc_dim      = 10240;
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

// A host page-cache view over the streaming (validate-only) artifact objects. Reads use pread so
// the OS page cache serves repeated rows without a large resident mapping. Only the gathered rows
// are transferred to device.
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

    std::uint64_t file_bytes = 0;

    // token embedding [vocab, hidden] BF16
    std::uint64_t token_emb_off       = 0;
    std::uint32_t token_emb_row_bytes = 0;
    // PLE table [ple_rows, ple_dim] BF16
    std::uint64_t ple_off       = 0;
    std::uint32_t ple_row_bytes = 0;

    // PLE descriptors. Each stored I32 u64 pair is little-endian (lo, hi). The artifact carries
    // 3 u64 n-gram multipliers and 16 u64 head offsets / 16 u64 head vocab sizes.
    std::array<std::uint64_t, 3> ple_multipliers{};
    std::array<std::uint64_t, 16> ple_head_offsets{};
    std::array<std::uint64_t, 16> ple_head_vocab_sizes{};

    // Routed expert banks, per layer. gate_up NVFP4 [655360,2560], down Q6 [1310720,640].
    struct RoutedSpan {
        std::uint64_t gate_up_off   = 0;
        std::uint64_t gate_up_bytes = 0;
        std::uint64_t down_off      = 0;
        std::uint64_t down_bytes    = 0;
    };
    std::array<RoutedSpan, layers> routed{};

    // Owned fd (O_RDONLY) for the pread-based streaming reads. Package-internal.
    int fd = -1;
};

struct ModelView {
    HostArtifactView host;
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
};

} // namespace detail

} // namespace ninfer::targets::qwen4exp
