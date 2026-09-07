#include "targets/qwen4exp/impl/load/bindings.h"

#include "targets/qwen4exp/impl/config.h"

#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen4exp/model_view.h>

#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <stdexcept>

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

// Binds one tensor and returns its handle. `placement` selects device / validate-only.
ObjectHandle bind_tensor(Binder& binder, const std::string& name, NumericFormat fmt,
                         StorageLayout layout, std::vector<std::uint64_t> shape,
                         TensorPlacement placement) {
    const ObjectHandle h =
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

// Reads a little-endian I32 pair-array from a payload span and lifts it to u64 values.
template <std::size_t N>
void lift_u64_pairs(const std::span<const std::byte>& data, std::array<std::uint64_t, N>& out) {
    if (data.size() < N * 2 * sizeof(std::int32_t)) {
        throw std::system_error(std::make_error_code(std::errc::invalid_argument),
                                "PLE descriptor span is shorter than expected");
    }
    const auto* v = reinterpret_cast<const std::int32_t*>(data.data());
    for (std::size_t i = 0; i < N; ++i) { out[i] = u64_from_i32_lo_hi(v + i * 2); }
}

// Fills the base/scale planes of one MoE bank (gate_up + down) from the object handles.
void fill_moe_bank(const Binder& binder, const ObjectHandle gu, const ObjectHandle dn,
                   MoeCode code, MoEBank& bank) {
    const auto split = [&](const ObjectHandle h, BankPlane& base,
                           BankPlane& scales) {
        const auto& td = std::get<artifact::TensorDescriptor>(binder.descriptor(h));
        const auto span = binder.payload(h);
        const auto shape2 = std::array<std::uint64_t, 2>{td.shape[0], td.shape[1]};
        const auto geo = artifact::row_split_geometry(td.format, shape2);
        if (geo.encoded_bytes != td.bytes) {
            throw std::system_error(std::make_error_code(std::errc::result_out_of_range),
                                    "row-split geometry does not match the stored object bytes for " +
                                        td.name);
        }
        base.offset = span.absolute_offset;
        base.bytes = geo.low_plane_bytes;
        scales.offset = span.absolute_offset + geo.scale_plane_offset;
        scales.bytes = geo.scale_plane_bytes;
    };
    bank.code = code;
    split(gu, bank.gate_up_base, bank.gate_up_scales);
    split(dn, bank.down_base, bank.down_scales);
}

} // namespace

// Resolves the device ObjectHandles into the GPU weight views (JM4b load-side).
DeviceWeights build_device_weights(const artifact::MaterializedArtifact& materialized,
                                   const DeviceHandleTable& h) {
    using namespace cfg;
    using artifact::NumericFormat;
    DeviceWeights w;
    for (std::uint32_t l = 0; l < layers; ++l) {
        const auto& H = h.layers[l];
        auto& W = w.per_layer[l];
        W.gdn.query_key = artifact::materialized_weight(materialized, H.gdn.query_key,
                                                        NumericFormat::W8G32_F16S, gdn_qk_proj_rows,
                                                        hidden);
        W.gdn.value_z = artifact::materialized_weight(materialized, H.gdn.value_z,
                                                      NumericFormat::W8G32_F16S, gdn_vz_proj_rows,
                                                      hidden);
        W.gdn.a_b_projection = artifact::materialized_tensor(
            materialized, H.gdn.a_b, NumericFormat::BF16, {gdn_ab_rows, hidden});
        W.gdn.a_log = artifact::materialized_tensor(materialized, H.gdn.a_log,
                                                    NumericFormat::FP32, {gdn_a_rows});
        W.gdn.dt_bias = artifact::materialized_tensor(materialized, H.gdn.dt_bias,
                                                      NumericFormat::FP32, {gdn_b_rows});
        W.gdn.convolution = artifact::materialized_tensor(materialized, H.gdn.conv,
                                                          NumericFormat::FP32,
                                                          {gdn_conv_width, gdn_conv_rows});
        W.gdn.norm = artifact::materialized_tensor(materialized, H.gdn.norm,
                                                   NumericFormat::FP32, {gdn_norm_rows});
        W.gdn.output = artifact::materialized_weight(materialized, H.gdn.output,
                                                     NumericFormat::W8G32_F16S, gdn_out_rows,
                                                     gdn_out_cols);
        W.qsa.query_key_gate_value = artifact::materialized_weight(
            materialized, H.qsa.qkgv, NumericFormat::W8G32_F16S, qkgv_rows, hidden);
        W.qsa.output = artifact::materialized_weight(materialized, H.qsa.output,
                                                     NumericFormat::W8G32_F16S, attn_out_rows,
                                                     attn_out_cols);
        W.qsa.query_norm = artifact::materialized_tensor(materialized, H.qsa.query_norm,
                                                         NumericFormat::FP32, {qsa_head_dim});
        W.qsa.key_norm = artifact::materialized_tensor(materialized, H.qsa.key_norm,
                                                       NumericFormat::FP32, {qsa_head_dim});
        W.qsa.indexer_query_proj = artifact::materialized_tensor(
            materialized, H.qsa.idx_query_proj, NumericFormat::BF16, {idx_q_proj, hidden});
        W.qsa.indexer_key_proj = artifact::materialized_tensor(
            materialized, H.qsa.idx_key_proj, NumericFormat::BF16, {idx_k_dim, hidden});
        W.qsa.indexer_query_norm = artifact::materialized_tensor(
            materialized, H.qsa.idx_query_norm, NumericFormat::FP32, {idx_k_dim});
        W.qsa.indexer_key_norm = artifact::materialized_tensor(materialized, H.qsa.idx_key_norm,
                                                               NumericFormat::FP32, {idx_k_dim});
        auto fill_hc = [&](DeviceWeights::Hc& ws, const DeviceHandleTable::Hc& hs) {
            ws.up = artifact::materialized_tensor(materialized, hs.up, NumericFormat::BF16,
                                                  {hc_dim, hc_lr});
            ws.down = artifact::materialized_tensor(materialized, hs.down, NumericFormat::BF16,
                                                    {hc_lr, hc_dim});
            ws.norm = artifact::materialized_tensor(materialized, hs.norm, NumericFormat::FP32,
                                                    {hc_dim});
            ws.inject = artifact::materialized_tensor(materialized, hs.inject, NumericFormat::BF16,
                                                      {hc_streams, hc_dim});
        };
        fill_hc(W.attn_hc, H.attn_hc);
        fill_hc(W.ffn_hc, H.ffn_hc);
        W.router = artifact::materialized_tensor(materialized, H.router, NumericFormat::FP32,
                                                 {moe_router_rows, hidden});
    }
    w.output_head = artifact::materialized_tensor(materialized, h.output_head,
                                                  NumericFormat::BF16, {vocab, hidden});
    w.output_hc.up = artifact::materialized_tensor(materialized, h.output_hc_up,
                                                   NumericFormat::BF16, {hc_dim, hc_lr});
    w.output_hc.down = artifact::materialized_tensor(materialized, h.output_hc_down,
                                                     NumericFormat::BF16, {hc_lr, hc_dim});
    w.output_hc.norm = artifact::materialized_tensor(materialized, h.output_hc_norm,
                                                     NumericFormat::FP32, {hc_dim});
    w.ple_key = artifact::materialized_tensor(materialized, h.ple_key, NumericFormat::BF16,
                                              {ple_key_rows, hidden});
    w.ple_value = artifact::materialized_tensor(materialized, h.ple_value, NumericFormat::BF16,
                                                {ple_value_rows, hidden});
    w.ple_conv = artifact::materialized_tensor(materialized, h.ple_conv, NumericFormat::BF16,
                                               {gdn_conv_width, ple_conv_rows});
    w.ple_norm_query = artifact::materialized_tensor(materialized, h.ple_norm_query,
                                                     NumericFormat::FP32, {ple_conv_rows});
    w.ple_norm_key = artifact::materialized_tensor(materialized, h.ple_norm_key,
                                                   NumericFormat::FP32, {ple_conv_rows});
    w.ple_norm_conv = artifact::materialized_tensor(materialized, h.ple_norm_conv,
                                                    NumericFormat::FP32, {ple_conv_rows});
    return w;
}

BindBundle bind_artifact(artifact::Binder& binder, const std::filesystem::path& path) {
    using namespace cfg;

    // --- identity: the package's resolve_weights has already validated the jbq4 identity ---

    // --- 6 frontend resources (retained on host) ---
    const qwen3_6::FrontendResourcePlan frontend = qwen3_6::bind_frontend_resources(binder);

    HostArtifactView view;
    DeviceHandleTable handles;

    // --- root tensors ---
    const ObjectHandle tok_emb = bind_tensor(
        binder, "text/token_embedding", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
        {static_cast<std::uint64_t>(vocab), hidden}, TensorPlacement::ValidateOnly);
    handles.output_head =
        bind_tensor(binder, "text/output_head", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                    {static_cast<std::uint64_t>(vocab), hidden}, TensorPlacement::Device);
    const ObjectHandle ple_table =
        bind_tensor(binder, "text/per_layer_token_embedding", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1,
                    {static_cast<std::uint64_t>(ple_rows), ple_dim},
                    TensorPlacement::ValidateOnly);
    handles.output_hc_up =
        bind_tensor(binder, "text/output_hc/up", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                    {hc_dim, hc_lr}, TensorPlacement::Device);
    handles.output_hc_down =
        bind_tensor(binder, "text/output_hc/down", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {hc_lr, hc_dim}, TensorPlacement::Device);
    handles.output_hc_norm = bind_tensor(binder, "text/output_hc/norm", NumericFormat::FP32,
                                                StorageLayout::ContiguousLeV1, {hc_dim},
                                                TensorPlacement::Device);
    const ObjectHandle ple_mult = bind_tensor(binder, "text/ple/layer_multipliers",
                                              NumericFormat::I32, StorageLayout::ContiguousLeV1,
                                              {ple_multipliers}, TensorPlacement::ValidateOnly);
    const ObjectHandle ple_off = bind_tensor(binder, "text/ple/head_offsets",
                                             NumericFormat::I32, StorageLayout::ContiguousLeV1,
                                             {32}, TensorPlacement::ValidateOnly);
    const ObjectHandle ple_size = bind_tensor(binder, "text/ple/head_vocab_sizes",
                                              NumericFormat::I32, StorageLayout::ContiguousLeV1,
                                              {32}, TensorPlacement::ValidateOnly);
    // PLE dense projections (shared across PLE layers).
    handles.ple_key = bind_tensor(binder, "text/ple/key", NumericFormat::BF16,
                                         StorageLayout::ContiguousLeV1, {ple_key_rows, hidden},
                                         TensorPlacement::Device);
    handles.ple_value = bind_tensor(binder, "text/ple/value", NumericFormat::BF16,
                                           StorageLayout::ContiguousLeV1,
                                           {ple_value_rows, hidden}, TensorPlacement::Device);
    handles.ple_conv = bind_tensor(binder, "text/ple/convolution", NumericFormat::BF16,
                                          StorageLayout::ContiguousLeV1,
                                          {gdn_conv_width, ple_conv_rows}, TensorPlacement::Device);
    handles.ple_norm_query = bind_tensor(binder, "text/ple/norm_query", NumericFormat::FP32,
                                                StorageLayout::ContiguousLeV1, {ple_conv_rows},
                                                TensorPlacement::Device);
    handles.ple_norm_key = bind_tensor(binder, "text/ple/norm_key", NumericFormat::FP32,
                                              StorageLayout::ContiguousLeV1, {ple_conv_rows},
                                              TensorPlacement::Device);
    handles.ple_norm_conv = bind_tensor(binder, "text/ple/norm_conv", NumericFormat::FP32,
                                               StorageLayout::ContiguousLeV1, {ple_conv_rows},
                                               TensorPlacement::Device);

    // --- per-layer tensors (48 x 21 objects) ---
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string L = "text/layers/" + std::to_string(layer);
        if (!is_full_attn_layer(layer)) {
            // GDN layer.
            auto& g = handles.layers[layer].gdn;
            g.query_key =
                bind_tensor(binder, L + "/gdn/query_key", NumericFormat::W8G32_F16S,
                            StorageLayout::RowSplitK128V1, {gdn_qk_proj_rows, hidden},
                            TensorPlacement::Device);
            g.value_z = bind_tensor(binder, L + "/gdn/value_z", NumericFormat::W8G32_F16S,
                                    StorageLayout::RowSplitK128V1, {gdn_vz_proj_rows, hidden},
                                    TensorPlacement::Device);
            g.a_b = bind_tensor(binder, L + "/gdn/a_b_projection", NumericFormat::BF16,
                                StorageLayout::ContiguousLeV1, {gdn_ab_rows, hidden},
                                TensorPlacement::Device);
            g.a_log = bind_tensor(binder, L + "/gdn/a_log", NumericFormat::FP32,
                                  StorageLayout::ContiguousLeV1, {gdn_a_rows},
                                  TensorPlacement::Device);
            g.dt_bias = bind_tensor(binder, L + "/gdn/dt_bias", NumericFormat::FP32,
                                    StorageLayout::ContiguousLeV1, {gdn_b_rows},
                                    TensorPlacement::Device);
            g.conv = bind_tensor(binder, L + "/gdn/convolution", NumericFormat::FP32,
                                 StorageLayout::ContiguousLeV1, {gdn_conv_width, gdn_conv_rows},
                                 TensorPlacement::Device);
            g.norm = bind_tensor(binder, L + "/gdn/norm", NumericFormat::FP32,
                                 StorageLayout::ContiguousLeV1, {gdn_norm_rows},
                                 TensorPlacement::Device);
            g.output = bind_tensor(binder, L + "/gdn/output", NumericFormat::W8G32_F16S,
                                   StorageLayout::RowSplitK128V1, {gdn_out_rows, gdn_out_cols},
                                   TensorPlacement::Device);
        } else {
            // Full-attn (QSA) layer.
            auto& q = handles.layers[layer].qsa;
            q.qkgv = bind_tensor(binder, L + "/attention/query_key_gate_value",
                                 NumericFormat::W8G32_F16S, StorageLayout::RowSplitK128V1,
                                 {qkgv_rows, hidden}, TensorPlacement::Device);
            q.output = bind_tensor(binder, L + "/attention/output", NumericFormat::W8G32_F16S,
                                   StorageLayout::RowSplitK128V1, {attn_out_rows, attn_out_cols},
                                   TensorPlacement::Device);
            q.query_norm =
                bind_tensor(binder, L + "/attention/query_norm", NumericFormat::FP32,
                            StorageLayout::ContiguousLeV1, {qsa_head_dim},
                            TensorPlacement::Device);
            q.key_norm = bind_tensor(binder, L + "/attention/key_norm", NumericFormat::FP32,
                                     StorageLayout::ContiguousLeV1, {qsa_head_dim},
                                     TensorPlacement::Device);
            q.idx_query_proj = bind_tensor(binder, L + "/indexer/query_proj", NumericFormat::BF16,
                                           StorageLayout::ContiguousLeV1, {idx_q_proj, hidden},
                                           TensorPlacement::Device);
            q.idx_key_proj = bind_tensor(binder, L + "/indexer/key_proj", NumericFormat::BF16,
                                         StorageLayout::ContiguousLeV1, {idx_k_dim, hidden},
                                         TensorPlacement::Device);
            q.idx_query_norm =
                bind_tensor(binder, L + "/indexer/query_norm", NumericFormat::FP32,
                            StorageLayout::ContiguousLeV1, {idx_k_dim}, TensorPlacement::Device);
            q.idx_key_norm = bind_tensor(binder, L + "/indexer/key_norm", NumericFormat::FP32,
                                         StorageLayout::ContiguousLeV1, {idx_k_dim},
                                         TensorPlacement::Device);
        }

        // HC units (attn + ffn) for every layer.
        auto& ha = handles.layers[layer].attn_hc;
        ha.up = bind_tensor(binder, L + "/hc_attn/up", NumericFormat::BF16,
                            StorageLayout::ContiguousLeV1, {hc_dim, hc_lr},
                            TensorPlacement::Device);
        ha.down = bind_tensor(binder, L + "/hc_attn/down", NumericFormat::BF16,
                              StorageLayout::ContiguousLeV1, {hc_lr, hc_dim},
                              TensorPlacement::Device);
        ha.norm = bind_tensor(binder, L + "/hc_attn/norm", NumericFormat::FP32,
                              StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        ha.inject = bind_tensor(binder, L + "/hc_attn/inject", NumericFormat::BF16,
                                StorageLayout::ContiguousLeV1, {hc_streams, hc_dim},
                                TensorPlacement::Device);
        auto& hf = handles.layers[layer].ffn_hc;
        hf.up = bind_tensor(binder, L + "/hc_ffn/up", NumericFormat::BF16,
                            StorageLayout::ContiguousLeV1, {hc_dim, hc_lr},
                            TensorPlacement::Device);
        hf.down = bind_tensor(binder, L + "/hc_ffn/down", NumericFormat::BF16,
                              StorageLayout::ContiguousLeV1, {hc_lr, hc_dim},
                              TensorPlacement::Device);
        hf.norm = bind_tensor(binder, L + "/hc_ffn/norm", NumericFormat::FP32,
                              StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        hf.inject = bind_tensor(binder, L + "/hc_ffn/inject", NumericFormat::BF16,
                                StorageLayout::ContiguousLeV1, {hc_streams, hc_dim},
                                TensorPlacement::Device);

        // MoE units for every layer. The codec follows the D14 map (Q8 for layers 0/1/47).
        const bool q8 = moe_layer_is_q8(layer);
        const NumericFormat moe_fmt = q8 ? NumericFormat::Q8G64_F16S : NumericFormat::Q4G64_F16S;
        const MoeCode code = q8 ? MoeCode::Q8G64 : MoeCode::Q4G64;
        const ObjectHandle gu =
            bind_tensor(binder, L + "/moe/routed_gate_up", moe_fmt, StorageLayout::RowSplitK128V1,
                        {routed_gu_rows, hidden}, TensorPlacement::ValidateOnly);
        const ObjectHandle dn =
            bind_tensor(binder, L + "/moe/routed_down", moe_fmt, StorageLayout::RowSplitK128V1,
                        {routed_dn_rows, moe_ffn}, TensorPlacement::ValidateOnly);
        const ObjectHandle router = bind_tensor(binder, L + "/moe/router_gate",
                                                NumericFormat::FP32,
                                                StorageLayout::ContiguousLeV1,
                                                {moe_router_rows, hidden}, TensorPlacement::Device);
        handles.layers[layer].router = router;
        const ObjectHandle sg =
            bind_tensor(binder, L + "/moe/shared_gate_up", moe_fmt,
                        StorageLayout::RowSplitK128V1, {shared_gu_rows, hidden},
                        TensorPlacement::ValidateOnly);
        const ObjectHandle sd = bind_tensor(binder, L + "/moe/shared_down", moe_fmt,
                                            StorageLayout::RowSplitK128V1, {shared_dn_rows, moe_ffn},
                                            TensorPlacement::ValidateOnly);

        // Bank spans for the page-cache MoE views.
        fill_moe_bank(binder, gu, dn, code, view.routed[layer]);
        view.moe_side[layer].router.offset =
            binder.payload(router).absolute_offset;
        view.moe_side[layer].router.bytes =
            static_cast<std::uint64_t>(moe_router_rows) * hidden * 4U;
        fill_moe_bank(binder, sg, sd, code, view.moe_side[layer].shared);
        view.layer_present[layer] = true;
    }

    // --- MTP (28 objects; Q8 MoE) ---
    {
        const std::string M = "mtp/";
        bind_tensor(binder, M + "fc_embedding", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {mtp_fc_rows, hidden},
                    TensorPlacement::Device);
        bind_tensor(binder, M + "fc_hidden", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {mtp_fc_rows, hidden},
                    TensorPlacement::Device);
        bind_tensor(binder, M + "pre_fc_norm_embedding", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {mtp_fc_rows}, TensorPlacement::Device);
        bind_tensor(binder, M + "pre_fc_norm_hidden", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, M + "hc/input_mix_weight_up", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {hc_dim, hc_lr}, TensorPlacement::Device);
        bind_tensor(binder, M + "hc/input_mix_weight_down", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {hc_lr, hc_dim}, TensorPlacement::Device);
        bind_tensor(binder, M + "hc/norm", NumericFormat::BF16, StorageLayout::ContiguousLeV1,
                    {hc_dim}, TensorPlacement::Device);
        for (const char* unit : {"layer/attn_hc", "layer/mlp_hc"}) {
            bind_tensor(binder, M + std::string(unit) + "/block_inject", NumericFormat::W8G32_F16S,
                        StorageLayout::RowSplitK128V1, {hc_streams, hc_dim},
                        TensorPlacement::Device);
            bind_tensor(binder, M + std::string(unit) + "/input_mix_weight_up",
                        NumericFormat::W8G32_F16S, StorageLayout::RowSplitK128V1, {hc_dim, hc_lr},
                        TensorPlacement::Device);
            bind_tensor(binder, M + std::string(unit) + "/input_mix_weight_down",
                        NumericFormat::W8G32_F16S, StorageLayout::RowSplitK128V1, {hc_lr, hc_dim},
                        TensorPlacement::Device);
            bind_tensor(binder, M + std::string(unit) + "/hc_norm", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {hc_dim}, TensorPlacement::Device);
        }
        bind_tensor(binder, M + "layer/attention/query_key_gate_value", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {mtp_qkgv_rows, hidden},
                    TensorPlacement::Device);
        bind_tensor(binder, M + "layer/attention/query_norm", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {qsa_head_dim}, TensorPlacement::Device);
        bind_tensor(binder, M + "layer/attention/key_norm", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {qsa_head_dim}, TensorPlacement::Device);
        bind_tensor(binder, M + "layer/attention/output", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {mtp_attn_out_rows, mtp_attn_out_cols},
                    TensorPlacement::Device);
        bind_tensor(binder, M + "layer/indexer/query_proj", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {mtp_idx_q_rows, hidden},
                    TensorPlacement::Device);
        bind_tensor(binder, M + "layer/indexer/key_proj", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {mtp_idx_k_rows, hidden},
                    TensorPlacement::Device);
        bind_tensor(binder, M + "layer/indexer/query_norm", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {idx_k_dim}, TensorPlacement::Device);
        bind_tensor(binder, M + "layer/indexer/key_norm", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {idx_k_dim}, TensorPlacement::Device);

        const ObjectHandle m_router = bind_tensor(binder, M + "moe/router_gate",
                                                  NumericFormat::FP32,
                                                  StorageLayout::ContiguousLeV1,
                                                  {moe_router_rows, hidden},
                                                  TensorPlacement::Device);
        const ObjectHandle m_gu =
            bind_tensor(binder, M + "moe/routed_gate_up", NumericFormat::Q8G64_F16S,
                        StorageLayout::RowSplitK128V1, {routed_gu_rows, hidden},
                        TensorPlacement::ValidateOnly);
        const ObjectHandle m_dn =
            bind_tensor(binder, M + "moe/routed_down", NumericFormat::Q8G64_F16S,
                        StorageLayout::RowSplitK128V1, {routed_dn_rows, moe_ffn},
                        TensorPlacement::ValidateOnly);
        const ObjectHandle m_sg =
            bind_tensor(binder, M + "moe/shared_gate_up", NumericFormat::Q8G64_F16S,
                        StorageLayout::RowSplitK128V1, {shared_gu_rows, hidden},
                        TensorPlacement::ValidateOnly);
        const ObjectHandle m_sd =
            bind_tensor(binder, M + "moe/shared_down", NumericFormat::Q8G64_F16S,
                        StorageLayout::RowSplitK128V1, {shared_dn_rows, moe_ffn},
                        TensorPlacement::ValidateOnly);
        view.mtp_moe_side.router.offset = binder.payload(m_router).absolute_offset;
        view.mtp_moe_side.router.bytes = static_cast<std::uint64_t>(moe_router_rows) * hidden * 4U;
        fill_moe_bank(binder, m_gu, m_dn, MoeCode::Q8G64, view.mtp_routed);
        fill_moe_bank(binder, m_sg, m_sd, MoeCode::Q8G64, view.mtp_moe_side.shared);
        view.mtp_present = true;
    }

    // --- vision (333 objects; validate-only) ---
    {
        const std::string V = "vision/";
        bind_tensor(binder, V + "patch_embedding", NumericFormat::Q6G64_F16S,
                    StorageLayout::RowSplitK128V1, {vision_hidden, vision_patch_input},
                    TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "patch_embedding_bias", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {vision_hidden},
                    TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "position_embedding", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {vision_pos_rows, vision_hidden},
                    TensorPlacement::ValidateOnly);
        for (std::uint32_t b = 0; b < vision_blocks; ++b) {
            const std::string B = V + "layers/" + std::to_string(b) + "/";
            bind_tensor(binder, B + "attention/qkv", NumericFormat::Q4G64_F16S,
                        StorageLayout::RowSplitK128V1, {3U * vision_hidden, vision_hidden},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "attention/qkv_bias", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {3U * vision_hidden},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "attention/output", NumericFormat::Q5G64_F16S,
                        StorageLayout::RowSplitK128V1, {vision_hidden, vision_hidden},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "attention/output_bias", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {vision_hidden},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "mlp/fc1", NumericFormat::Q4G64_F16S,
                        StorageLayout::RowSplitK128V1, {vision_mlp_intermediate, vision_hidden},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "mlp/fc1_bias", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {vision_mlp_intermediate},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "mlp/fc2", NumericFormat::Q5G64_F16S,
                        StorageLayout::RowSplitK128V1, {vision_hidden, vision_mlp_intermediate},
                        TensorPlacement::ValidateOnly);
            bind_tensor(binder, B + "mlp/fc2_bias", NumericFormat::BF16,
                        StorageLayout::ContiguousLeV1, {vision_hidden},
                        TensorPlacement::ValidateOnly);
            for (const char* norm : {"norm1", "norm2"}) {
                bind_tensor(binder, B + std::string(norm) + "/weight", NumericFormat::BF16,
                            StorageLayout::ContiguousLeV1, {vision_hidden},
                            TensorPlacement::ValidateOnly);
                bind_tensor(binder, B + std::string(norm) + "/bias", NumericFormat::BF16,
                            StorageLayout::ContiguousLeV1, {vision_hidden},
                            TensorPlacement::ValidateOnly);
            }
        }
        bind_tensor(binder, V + "merger/fc1", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {vision_merger_intermediate,
                                                   vision_merger_intermediate},
                    TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "merger/fc1_bias", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {vision_merger_intermediate},
                    TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "merger/fc2", NumericFormat::W8G32_F16S,
                    StorageLayout::RowSplitK128V1, {hidden, vision_merger_intermediate},
                    TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "merger/fc2_bias", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {hidden}, TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "merger/norm/weight", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {vision_hidden}, TensorPlacement::ValidateOnly);
        bind_tensor(binder, V + "merger/norm/bias", NumericFormat::BF16,
                    StorageLayout::ContiguousLeV1, {vision_hidden}, TensorPlacement::ValidateOnly);
        view.vision_present = true;
    }

    // --- open the host view fd, map the file, lift absolute offsets + PLE descriptors ---
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "failed to open artifact for the host streaming view: " +
                                    path.string());
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::system_error(errno, std::generic_category(), "fstat of the artifact failed");
    }
    view.file_bytes = static_cast<std::uint64_t>(st.st_size);
    void* map_base = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                            MAP_PRIVATE, fd, 0);
    if (map_base == MAP_FAILED) {
        ::close(fd);
        throw std::system_error(errno, std::generic_category(),
                                "mmap of the artifact failed for the MoE bank views");
    }
    view.map_base = map_base;
    view.map_bytes = static_cast<std::uint64_t>(st.st_size);

    // Payload origin: absolute_offset - object-relative offset (from any tensor descriptor).
    const auto& tok_desc = std::get<artifact::TensorDescriptor>(binder.descriptor(tok_emb));
    const auto tok_span = binder.payload(tok_emb);
    view.payload_off = tok_span.absolute_offset - tok_desc.offset;

    view.token_emb_off = tok_span.absolute_offset;
    view.token_emb_row_bytes = static_cast<std::uint32_t>(hidden * 2U);
    view.ple_off = binder.payload(ple_table).absolute_offset;
    view.ple_row_bytes = static_cast<std::uint32_t>(ple_dim * 2U);

    lift_u64_pairs(binder.payload(ple_mult).data, view.ple_multipliers);
    lift_u64_pairs(binder.payload(ple_off).data, view.ple_head_offsets);
    lift_u64_pairs(binder.payload(ple_size).data, view.ple_head_vocab_sizes);

    view.fd = fd;
    return BindBundle{std::move(view), std::move(handles), std::move(frontend)};
}

// HostArtifactView method definitions (declared in model_view.h; read_at is in cpu.cpp).
const std::byte* HostArtifactView::map_ptr(std::uint64_t off) const noexcept {
    return static_cast<const std::byte*>(map_base) + off;
}

} // namespace ninfer::targets::qwen4exp::detail
