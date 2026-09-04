#include "targets/qwen4exp/impl/cpu/cpu.h"

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen4exp/impl/cpu/ple_hash.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <sys/mman.h>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {
// ---- HostArtifactView special members ----
HostArtifactView::HostArtifactView(HostArtifactView&& other) noexcept {
    *this = std::move(other);
}

HostArtifactView& HostArtifactView::operator=(HostArtifactView&& other) noexcept {
    if (this != &other) {
        if (fd >= 0) { ::close(fd); }
        if (map_base != nullptr) { ::munmap(map_base, static_cast<std::size_t>(map_bytes)); }
        token_emb_off       = other.token_emb_off;
        token_emb_row_bytes = other.token_emb_row_bytes;
        ple_off             = other.ple_off;
        ple_row_bytes       = other.ple_row_bytes;
        ple_multipliers     = other.ple_multipliers;
        ple_head_offsets    = other.ple_head_offsets;
        ple_head_vocab_sizes = other.ple_head_vocab_sizes;
        layer_present       = other.layer_present;
        mtp_present         = other.mtp_present;
        vision_present      = other.vision_present;
        routed              = other.routed;
        mtp_routed          = other.mtp_routed;
        moe_side            = other.moe_side;
        mtp_moe_side        = other.mtp_moe_side;
        file_bytes          = other.file_bytes;
        fd                  = other.fd;
        map_base            = other.map_base;
        map_bytes           = other.map_bytes;
        payload_off         = other.payload_off;
        other.fd            = -1;
        other.map_base      = nullptr;
        other.map_bytes     = 0;
    }
    return *this;
}

HostArtifactView::~HostArtifactView() {
    if (fd >= 0) { ::close(fd); }
    if (map_base != nullptr) { ::munmap(map_base, static_cast<std::size_t>(map_bytes)); }
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

} // namespace ninfer::targets::qwen4exp::detail
