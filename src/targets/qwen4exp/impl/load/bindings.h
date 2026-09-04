#pragma once

// qwen4exp (jbq4) artifact binding: validates all 1,390 objects (1,384 tensors + 6 resources)
// against the jbq4 geometry (D14 MoE codec map, MTP, vision), decides the device /
// validate-only split, lifts the small PLE descriptors to host, and records the page-cache
// MoE bank spans (routed + shared + router) in the HostArtifactView.

#include "ninfer/types.h"
#include <ninfer/targets/qwen4exp/model_view.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"

#include <filesystem>

namespace ninfer::targets::qwen4exp::detail {

using artifact::ObjectHandle;

// Device ObjectHandles for the 48 text layers, the root output head, and the shared PLE dense
// block, captured at bind time (pre-materialization) and resolved to device pointers in
// construct_loaded_model (JM4b load-side). Mirrors the DeviceWeights layout.
struct DeviceHandleTable {
    struct GdnBlock {
        ObjectHandle query_key, value_z, a_b, a_log, dt_bias, conv, norm, output;
    };
    struct QsaBlock {
        ObjectHandle qkgv, output, query_norm, key_norm, idx_query_proj, idx_key_proj,
            idx_query_norm, idx_key_norm;
    };
    struct Hc {
        ObjectHandle up, down, norm, inject;
    };
    struct Layer {
        GdnBlock gdn;
        QsaBlock qsa;
        Hc       attn_hc;
        Hc       ffn_hc;
        ObjectHandle router;
    };
    std::array<Layer, 48> layers{};
    ObjectHandle output_head, output_hc_up, output_hc_down, output_hc_norm, output_hc_inject;
    ObjectHandle ple_key, ple_value, ple_conv, ple_norm_query, ple_norm_key, ple_norm_conv;
};

// Bundle returned by the binder: the host page-cache view (offsets, MoE bank spans, PLE
// descriptors, fd + MAP_PRIVATE mapping), the frontend resource handles, and the device
// ObjectHandles (resolved to device pointers post-materialization).
struct BindBundle {
    HostArtifactView    host;
    DeviceHandleTable   handles;
    qwen3_6::FrontendResourcePlan frontend;
};

// Binds every object. `path` is the artifact (used only to open the host view fd). Returns the
// populated host streaming view + the frontend resource plan; the caller (package.cpp) then calls
// binder.finish() for the materialization plan.
BindBundle bind_artifact(artifact::Binder& binder, const std::filesystem::path& path);

// Resolves the device ObjectHandles captured at bind time into the GPU weight views
// (JM4b load-side). Called post-materialization from construct_loaded_model.
DeviceWeights build_device_weights(const artifact::MaterializedArtifact& materialized,
                                   const DeviceHandleTable& handles);

} // namespace ninfer::targets::qwen4exp::detail
