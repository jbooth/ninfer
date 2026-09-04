// jbq4 binding gate (JM4a): binds the real artifact through the package's public load path
// (plan_load -> bind_artifact), verifies the full object table (1,384 tensors + 6 frontend
// resources = 1,390), the D14 MoE codec map, the routed/shared bank plane extents, the MTP and
// vision presence, and the page-cache host view (mmap + lifted PLE descriptors).
//
// Host-only: no CUDA calls, no device materialization (the plan is built but never executed).
// Skips (exit 77) when the artifact is missing.

#include "artifact/binder.h"
#include "artifact/reader.h"

#include <ninfer/targets/qwen4exp/model_view.h>
#include <ninfer/targets/qwen4exp/package.h>
#include <ninfer/types.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

namespace {

using ninfer::artifact::Binder;
using ninfer::artifact::Reader;
using ninfer::ops::MoeCode;
using ninfer::EngineOptions;
using ninfer::targets::qwen4exp::Package;

const char kArtifactPath[] = "/llm/models/Qwen3.8-Flash-Next-JBQ4.ninfer";

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

} // namespace

int main() {
    const std::filesystem::path path(kArtifactPath);
    if (!std::filesystem::exists(path)) {
        std::printf("SKIP: artifact not found at %s\n", kArtifactPath);
        std::exit(77);
    }

    std::printf("jbq4 bind gate: %s\n", kArtifactPath);
    std::fflush(stdout);

    Reader reader(path);
    const auto& identity = reader.identity();
    check(std::string_view(identity.model_id) == Package::model_id, "identity.model_id");
    check(std::string_view(identity.weights_id) == Package::weights_id, "identity.weights_id");
    check(reader.objects().size() == 1390, "directory census is 1,390 objects");

    // resolve_weights must accept exactly this identity.
    const auto profile = Package::resolve_weights(identity);
    check(profile == Package::WeightsProfile::Qwen4expJbQ4, "resolve_weights returns JbQ4");

    // Bind the full table through the package's load path. A single wrong name/format/shape in
    // the binder throws; a successful plan therefore proves all 1,390 objects bound.
    Binder binder(reader);
    EngineOptions options;
    options.artifact_path = path;
    auto plan = Package::plan_load(binder, options, profile);
    const auto& view = plan.host_view();

    std::printf("  device objects: %zu, device capacity: %zu B\n",
                plan.materialization().device_objects.size(),
                plan.materialization().device_capacity_bytes);
    std::fflush(stdout);

    // All 48 layers present; the D14 codec map (Q8 for layers 0/1/47, Q4 otherwise). For each
    // codec the routed bank splits into a base (weight) plane + an FP16 scale plane:
    //   Q4: gu base 838,860,800 / dn base 419,430,400; scale 52,428,800 / 26,214,400
    //   Q8: gu base 1,677,721,600 / dn base 838,860,800; scale 52,428,800 / 26,214,400
    for (std::uint32_t l = 0; l < 48; ++l) {
        check(view.layer_present[l], "layer_present");
        const bool q8 = (l == 0 || l == 1 || l == 47);
        check(view.routed[l].code == (q8 ? MoeCode::Q8G64 : MoeCode::Q4G64), "routed codec (D14)");
        check(view.moe_side[l].shared.code == (q8 ? MoeCode::Q8G64 : MoeCode::Q4G64),
              "shared codec (D14)");
        const std::uint64_t gu_base = q8 ? 1677721600ULL : 838860800ULL;
        const std::uint64_t dn_base = q8 ? 838860800ULL : 419430400ULL;
        check(view.routed[l].gate_up_base.bytes == gu_base, "routed gate_up base bytes");
        check(view.routed[l].down_base.bytes == dn_base, "routed down base bytes");
        check(view.routed[l].gate_up_scales.bytes == 52428800ULL, "routed gate_up scale bytes");
        check(view.routed[l].down_scales.bytes == 26214400ULL, "routed down scale bytes");
    }
    check(view.mtp_present, "mtp_present");
    check(view.vision_present, "vision_present");
    check(view.mtp_routed.code == ninfer::ops::MoeCode::Q8G64, "mtp routed codec (Q8)");
    check(view.mtp_moe_side.router.bytes == 5253120ULL, "mtp router bytes");

    // Host view: the MAP_PRIVATE mapping covers the file; the PLE descriptors are lifted.
    check(view.map_bytes == static_cast<std::uint64_t>(std::filesystem::file_size(path)),
          "mmap covers the file");
    check(view.ple_head_offsets[0] == 0, "ple head 0 offset is 0");
    check(view.ple_multipliers.size() == 3, "ple multipliers count");
    check(view.ple_head_vocab_sizes.size() == 16, "ple head vocab sizes count");

    if (g_failures) {
        std::printf("jbq4 bind gate: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("jbq4 bind gate: OK (1,390 objects bound; D14 map, banks, host view verified)\n");
    return 0;
}
