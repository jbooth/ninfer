#pragma once

// qwen4exp artifact binding: validates all 1077 objects (1071 tensors + 6 resources) against the
// jbnvfp4 geometry, decides the device / host / validate-only split, and lifts the small PLE
// descriptors to host for the real CPU gather routines.

#include "ninfer/types.h"
#include <ninfer/targets/qwen4exp/model_view.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "artifact/binder.h"

#include <filesystem>

namespace ninfer::targets::qwen4exp::detail {

// Bundle returned by the binder: the host streaming view (offsets + PLE descriptors + fd) and the
// frontend resource handles (used later to take the resource payloads from the materialized
// artifact).
struct BindBundle {
    HostArtifactView host;
    qwen3_6::FrontendResourcePlan frontend;
};

// Binds every object. `path` is the artifact (used only to open the host view fd). Returns the
// populated host streaming view + the frontend resource plan; the caller (package.cpp) then calls
// binder.finish() for the materialization plan.
BindBundle bind_artifact(artifact::Binder& binder, const std::filesystem::path& path);

} // namespace ninfer::targets::qwen4exp::detail
