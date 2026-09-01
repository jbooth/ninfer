#include "targets/qwen4exp/impl/cpu/cpu.h"

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen4exp/impl/cpu/ple_hash.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {
// ---- HostArtifactView special members ----
HostArtifactView::HostArtifactView(HostArtifactView&& other) noexcept
    : token_emb_off(other.token_emb_off),
      token_emb_row_bytes(other.token_emb_row_bytes),
      ple_off(other.ple_off),
      ple_row_bytes(other.ple_row_bytes),
      ple_multipliers(other.ple_multipliers),
      ple_head_offsets(other.ple_head_offsets),
      ple_head_vocab_sizes(other.ple_head_vocab_sizes),
      routed(other.routed) {
    fd = other.fd;
    other.fd = -1;
}

HostArtifactView& HostArtifactView::operator=(HostArtifactView&& other) noexcept {
    if (this != &other) {
        if (fd >= 0) { ::close(fd); }
        token_emb_off = other.token_emb_off;
        token_emb_row_bytes = other.token_emb_row_bytes;
        ple_off = other.ple_off;
        ple_row_bytes = other.ple_row_bytes;
        ple_multipliers = other.ple_multipliers;
        ple_head_offsets = other.ple_head_offsets;
        ple_head_vocab_sizes = other.ple_head_vocab_sizes;
        routed = other.routed;
        fd = other.fd;
        other.fd = -1;
    }
    return *this;
}

HostArtifactView::~HostArtifactView() {
    if (fd >= 0) { ::close(fd); }
}

void HostArtifactView::read_at(std::uint64_t off, void* destination, std::size_t bytes) const {
    std::size_t total = 0;
    auto* p = static_cast<std::byte*>(destination);
    while (total < bytes) {
        const std::size_t n =
            ::pread(fd, p + total, bytes - total, static_cast<off_t>(off + total));
        if (n < 0) {
            if (errno == EINTR) { continue; }
            throw std::system_error(errno, std::generic_category(),
                                    "pread from the host artifact view failed");
        }
        if (n == 0) {
            throw std::system_error(std::make_error_code(std::errc::io_error),
                                    "short read from the host artifact view");
        }
        total += n;
    }
}

// ---- CPU gather / streaming routines ----
void gather_token_embedding(const HostArtifactView& art, const TokenId* ids, std::int32_t T,
                            std::byte* out_bf16, cudaStream_t stream) {
    std::vector<std::byte> host(static_cast<std::size_t>(T) * hidden * 2U);
    for (std::int32_t i = 0; i < T; ++i) {
        const std::int32_t token = ids[i];
        const std::uint64_t off = art.token_emb_off + static_cast<std::uint64_t>(token) * art.token_emb_row_bytes;
        art.read_at(off, host.data() + static_cast<std::size_t>(i) * hidden * 2U, hidden * 2U);
    }
    if (cudaMemcpyAsync(out_bf16, host.data(), host.size(), cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
        throw std::system_error(std::make_error_code(std::errc::io_error),
                                "H2D of the token embedding gather failed");
    }
}

void gather_ple_layer1(const HostArtifactView& art, const TokenId* ids, std::int32_t i, std::int32_t T,
                       std::byte* out_bf16, cudaStream_t stream) {
    // 16 heads, each a 160-dim row gathered from the PLE table. The row indices are the
    // reference n-gram hash (compute_ple_rows: EOS-padded predecessors, EOS window cut).
    // Row h of `out_bf16` receives the head-h table row. (void)T: position `i` fully
    // determines the hash; the sequence length is carried by the caller's ring.
    const auto rows = compute_ple_rows(i, ids, art.ple_multipliers, art.ple_head_offsets,
                                       art.ple_head_vocab_sizes, cfg::ple_eos);
    std::vector<std::byte> row(ple_dim * 2U);
    for (std::uint32_t h = 0; h < ple_n_heads; ++h) {
        const std::uint64_t off = art.ple_off + rows[h] * art.ple_row_bytes;
        art.read_at(off, row.data(), ple_dim * 2U);
        if (cudaMemcpyAsync(out_bf16 + static_cast<std::size_t>(h) * ple_dim * 2U, row.data(),
                            ple_dim * 2U, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::system_error(std::make_error_code(std::errc::io_error),
                                    "H2D of a PLE embedding row failed");
        }
    }
    (void)T;
}

void stream_experts(const HostArtifactView& art, std::int32_t layer, const TokenId* router_row_dst,
                    const std::uint8_t* selected_experts, std::int32_t topk, std::byte* gu_scratch,
                    std::byte* dn_scratch, cudaStream_t stream) {
    (void)router_row_dst;
    const HostArtifactView::RoutedSpan& bank = art.routed[layer];
    const std::size_t gu_slot = bank.gate_up_bytes / moe_experts;
    const std::size_t dn_slot = bank.down_bytes / moe_experts;
    for (std::int32_t k = 0; k < topk; ++k) {
        const std::uint32_t e = selected_experts[k];
        std::vector<std::byte> host_gu(gu_slot);
        std::vector<std::byte> host_dn(dn_slot);
        art.read_at(bank.gate_up_off + static_cast<std::uint64_t>(e) * gu_slot, host_gu.data(),
                    gu_slot);
        art.read_at(bank.down_off + static_cast<std::uint64_t>(e) * dn_slot, host_dn.data(), dn_slot);
        auto* gu_k = gu_scratch + static_cast<std::size_t>(k) * gu_slot;
        auto* dn_k = dn_scratch + static_cast<std::size_t>(k) * dn_slot;
        if (cudaMemcpyAsync(gu_k, host_gu.data(), gu_slot, cudaMemcpyHostToDevice, stream) !=
                cudaSuccess ||
            cudaMemcpyAsync(dn_k, host_dn.data(), dn_slot, cudaMemcpyHostToDevice, stream) !=
                cudaSuccess) {
            throw std::system_error(std::make_error_code(std::errc::io_error),
                                    "H2D of an expert slot failed");
        }
    }
}

} // namespace ninfer::targets::qwen4exp::detail
