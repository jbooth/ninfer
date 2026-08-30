#include "targets/qwen4exp/impl/load/bindings.h"

#include "targets/qwen4exp/impl/config.h"

#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {
namespace {

using artifact::Binder;
using artifact::NumericFormat;
using artifact::ObjectHandle;
using artifact::StorageLayout;
using artifact::TensorPlacement;

// Skeleton execution switch: when false, every tensor is bound validate-only so the device
// footprint is ~0 (the stub op leaves do not read device weights; only the host pread streams and
// the device sampler run). The real deliverable config keeps this true so the 6 GB dense residency
// is physically exercised for the <=32 GB gate.
constexpr bool kDeviceResident = true;

// Small "parameter" tensors (norms, biases, conv filters, PLE scalars) stay resident so the
// device arena is non-empty; only the large weight matrices drop to validate-only in skeleton mode.
constexpr std::uint64_t kResidentNumelThreshold = 100000;

// Binds one tensor and returns its handle. `placement` selects device / validate-only.
ObjectHandle bind_tensor(Binder& binder, const std::string& name, NumericFormat fmt,
                         StorageLayout layout, std::vector<std::uint64_t> shape,
                         TensorPlacement placement) {
    if (!kDeviceResident && placement == TensorPlacement::Device) {
        std::uint64_t numel = 1;
        for (std::uint64_t d : shape) { numel *= d; }
        if (numel >= kResidentNumelThreshold) { placement = TensorPlacement::ValidateOnly; }
    }
    ObjectHandle h =
        binder.require_tensor(name, fmt, layout, std::span<const std::uint64_t>(shape));
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(h);
    } else {
        binder.validate_only(h);
    }
    return h;
}

std::uint64_t u64_from_i32_lo_hi(const std::int32_t* v) noexcept {
    return static_cast<std::uint32_t>(v[0]) |
           (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v[1])) << 32);
}

// Reads a little-endian I32 pair-array from an mmap'd payload span and lifts it to u64 values.
template <std::size_t N>
void lift_u64_pairs(const std::span<const std::byte>& data, std::array<std::uint64_t, N>& out) {
    if (data.size() < N * 2 * sizeof(std::int32_t)) {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                "PLE descriptor span is shorter than expected");
    }
    const auto* v = reinterpret_cast<const std::int32_t*>(data.data());
    for (std::size_t i = 0; i < N; ++i) { out[i] = u64_from_i32_lo_hi(v + i * 2); }
}

} // namespace

BindBundle bind_artifact(artifact::Binder& binder, const std::filesystem::path& path) {
    using namespace cfg;

    // --- 6 frontend resources (retained on host) ---
    const qwen3_6::FrontendResourcePlan frontend = qwen3_6::bind_frontend_resources(binder);

    HostArtifactView view;

    // --- root tensors ---
    const ObjectHandle tok_emb = bind_tensor(
        binder, "text/token_embedding", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
        {static_cast<std::uint64_t>(vocab), hidden}, TensorPlacement::ValidateOnly);
    bind_tensor(binder, "text/output_head", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                {static_cast<std::uint64_t>(vocab), hidden}, TensorPlacement::Device);
    const ObjectHandle ple_table =
        bind_tensor(binder, "text/per_layer_token_embedding", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1,
                    {static_cast<std::uint64_t>(ple_rows), ple_dim},
                    TensorPlacement::ValidateOnly);
    bind_tensor(binder, "text/output_hc/up", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                {hc_dim, hc_lr}, TensorPlacement::Device);
    bind_tensor(binder, "text/output_hc/down", NumericFormat::BF16,
                StorageLayout::ContiguousLeV1, {hc_lr, hc_dim}, TensorPlacement::Device);
    bind_tensor(binder, "text/output_hc/norm", NumericFormat::FP32,
                StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
    const ObjectHandle ple_mult = bind_tensor(binder, "text/ple/layer_multipliers",
                                              NumericFormat::I32, StorageLayout::ContiguousLeV1,
                                              {3 * 2}, TensorPlacement::Device);
    const ObjectHandle ple_off = bind_tensor(binder, "text/ple/head_offsets",
                                             NumericFormat::I32, StorageLayout::ContiguousLeV1,
                                             {16 * 2}, TensorPlacement::Device);
    const ObjectHandle ple_size = bind_tensor(binder, "text/ple/head_vocab_sizes",
                                              NumericFormat::I32, StorageLayout::ContiguousLeV1,
                                              {16 * 2}, TensorPlacement::Device);

    // --- per-layer tensors ---
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string L = "text/layers/" + std::to_string(layer);
        if (layer % 4 != 3) {
            // GDN layer.
            bind_tensor(binder, L + "/gdn/query_key", NumericFormat::W8G32_F16S,
                        StorageLayout::RowSplitK128V1, {4096, hidden}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/value_z", NumericFormat::W8G32_F16S,
                        StorageLayout::RowSplitK128V1, {12288, hidden}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/a_b_projection", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {96, hidden}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/a_log", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {48}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/dt_bias", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {48}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/convolution", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {4, hc_dim}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/norm", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {128}, TensorPlacement::Device);
            bind_tensor(binder, L + "/gdn/output", NumericFormat::W8G32_F16S,
                        StorageLayout::RowSplitK128V1, {hidden, 6144}, TensorPlacement::Device);
        } else {
            // Full-attn (QSA) layer.
            bind_tensor(binder, L + "/attention/query_key_gate_value", NumericFormat::W8G32_F16S,
                        StorageLayout::RowSplitK128V1, {13312, hidden}, TensorPlacement::Device);
            bind_tensor(binder, L + "/attention/output", NumericFormat::W8G32_F16S,
                        StorageLayout::RowSplitK128V1, {hidden, 6144}, TensorPlacement::Device);
            bind_tensor(binder, L + "/attention/query_norm", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {256}, TensorPlacement::Device);
            bind_tensor(binder, L + "/attention/key_norm", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {256}, TensorPlacement::Device);
            bind_tensor(binder, L + "/indexer/query_proj", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {512, hidden}, TensorPlacement::Device);
            bind_tensor(binder, L + "/indexer/key_proj", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {128, hidden}, TensorPlacement::Device);
            bind_tensor(binder, L + "/indexer/query_norm", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {128}, TensorPlacement::Device);
            bind_tensor(binder, L + "/indexer/key_norm", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {128}, TensorPlacement::Device);
        }

        // HC units (attn + ffn) for every layer.
        bind_tensor(binder, L + "/hc_attn/up", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                    {hc_dim, hc_lr}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_attn/down", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {hc_lr, hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_attn/norm", NumericFormat::FP32,
                    StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_attn/inject", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {4, hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_ffn/up", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                    {hc_dim, hc_lr}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_ffn/down", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {hc_lr, hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_ffn/norm", NumericFormat::FP32,
                    StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, L + "/hc_ffn/inject", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {4, hc_dim}, TensorPlacement::Device);

        // MoE units for every layer.
        const ObjectHandle gu = bind_tensor(binder, L + "/moe/routed_gate_up", NumericFormat::NVFP4,
                                            StorageLayout::BlockScaleK16M128x4V1,
                                            {655360, hidden}, TensorPlacement::ValidateOnly);
        bind_tensor(binder, L + "/moe/gate_up_input_divisor", NumericFormat::FP32,
                    StorageLayout::ContiguousLeV1, {}, TensorPlacement::Device);
        const ObjectHandle dn = bind_tensor(binder, L + "/moe/routed_down", NumericFormat::Q6G64_F16S,
                                            StorageLayout::RowSplitK128V1, {1310720, 640},
                                            TensorPlacement::ValidateOnly);
        bind_tensor(binder, L + "/moe/router_gate", NumericFormat::FP32,
                    StorageLayout::ContiguousLeV1, {513, hidden}, TensorPlacement::Device);
        bind_tensor(binder, L + "/moe/shared_gate_up", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {1280, hidden}, TensorPlacement::Device);
        bind_tensor(binder, L + "/moe/shared_down", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {hidden, 640}, TensorPlacement::Device);

        // PLE projection weights live under the layer-1 unit but at the root name.
        if (layer == 1) {
            bind_tensor(binder, "text/ple/key", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {hc_dim, hidden}, TensorPlacement::Device);
            bind_tensor(binder, "text/ple/value", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {hidden, hidden}, TensorPlacement::Device);
            bind_tensor(binder, "text/ple/convolution", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {4, hc_dim}, TensorPlacement::Device);
            bind_tensor(binder, "text/ple/norm_query", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
            bind_tensor(binder, "text/ple/norm_key", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
            bind_tensor(binder, "text/ple/norm_conv", NumericFormat::FP32,
                        StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        }

        // Routed bank offsets.
        const auto gu_span = binder.payload(gu);
        const auto dn_span = binder.payload(dn);
        view.routed[layer].gate_up_off = gu_span.absolute_offset;
        view.routed[layer].gate_up_bytes = gu_span.data.size();
        view.routed[layer].down_off = dn_span.absolute_offset;
        view.routed[layer].down_bytes = dn_span.data.size();
    }

    // --- open the host view fd and lift absolute offsets + PLE descriptors ---
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "failed to open artifact for the host streaming view: " +
                                    path.string());
    }

    view.token_emb_off = binder.payload(tok_emb).absolute_offset;
    view.token_emb_row_bytes = static_cast<std::uint32_t>(hidden * 2U);
    view.ple_off = binder.payload(ple_table).absolute_offset;
    view.ple_row_bytes = static_cast<std::uint32_t>(ple_dim * 2U);

    lift_u64_pairs(binder.payload(ple_mult).data, view.ple_multipliers);
    lift_u64_pairs(binder.payload(ple_off).data, view.ple_head_offsets);
    lift_u64_pairs(binder.payload(ple_size).data, view.ple_head_vocab_sizes);

    view.fd = fd;
    return BindBundle{std::move(view), frontend};
}

} // namespace ninfer::targets::qwen4exp::detail
