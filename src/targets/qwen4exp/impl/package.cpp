// qwen4exp (Qwen3.8-Flash-Next / jbq4) target package: the seven static factories plus the
// LoadPlan / LoadedModel PIMPL bodies. Single-target, non-templated family; reuses the qwen3_6
// frontend.

#include <ninfer/targets/qwen4exp/package.h>

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include "targets/qwen4exp/impl/load/bindings.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen4exp::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, artifact::MaterializationPlan materialization_in,
         BindBundle bundle_in)
        : weights_profile(weights_profile_in), materialization(std::move(materialization_in)),
          bundle(std::move(bundle_in)) {}

    WeightsProfile weights_profile = WeightsProfile::Qwen4expJbQ4;
    artifact::MaterializationPlan materialization;
    BindBundle bundle;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept                = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept     = default;
LoadPlan::~LoadPlan()                                  = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->materialization;
}

const HostArtifactView& LoadPlan::host_view() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->bundle.host;
}

const qwen3_6::FrontendResourcePlan& LoadPlan::frontend_plan() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->bundle.frontend;
}

class LoadedModel::Impl {
public:
    explicit Impl(ModelView model_in) : model(std::move(model_in)) {}

    ModelView model;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel()                                   = default;

const ModelView& LoadedModel::view() const {
    if (impl_ == nullptr) { throw std::logic_error("loaded model is empty"); }
    return impl_->model;
}

} // namespace ninfer::targets::qwen4exp::detail

namespace ninfer::targets::qwen4exp {
namespace {

// General-task presets published with the exact model (matches the qwen3_6 family values).
constexpr ModelSamplingDefaults kDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

// KV geometry (page-group stride). The fixed per-sequence state is the GDN recurrent + conv state
// plus the PLE conv history; the main KV and indexer side cache grow per token.
constexpr std::uint32_t kMainPageTokens   = 128;
constexpr std::uint32_t kMinimumPageGroups = 1;
constexpr std::uint32_t kMaximumPageGroups = 4096;

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == weights_id) {
        return WeightsProfile::Qwen4expJbQ4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    detail::BindBundle bundle = detail::bind_artifact(binder, options.artifact_path);
    const artifact::MaterializationPlan materialization = binder.finish();
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, std::move(materialization), std::move(bundle)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }

    detail::ModelView model;
    // Move the host streaming view (owns the pread fd + MAP_PRIVATE mapping) into the model.
    model.host = std::move(plan.impl_->bundle.host);
    // Resolve the device ObjectHandles into the GPU weight views (JM4b load-side).
    model.weights = detail::build_device_weights(materialized, plan.impl_->bundle.handles);
    model.frontend = qwen3_6::take_frontend_resources(materialized, plan.impl_->bundle.frontend);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::make_unique<LoadedModel::Impl>(
        std::move(model))));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    const auto features = qwen3_6::startup_features(options);
    return qwen3_6::make_frontend(model.impl_->model.frontend, qwen3_6::FrontendOptions{
        .vision_enabled                = features.vision,
        .max_context                   = options.max_context,
        .media_cache_bytes             = options.media_cache_bytes,
        .media_live_bytes              = options.media_live_bytes,
        .media_preprocess_threads      = options.media_preprocess_threads,
        .max_cache_markers_per_request =
            options.context_cache.max_cache_markers_per_request.value_or(4),
    });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    (void)device;
    (void)weights_profile;
    const std::uint32_t max_concurrency =
        options.max_concurrency == 0 ? 1U : options.max_concurrency;

    // Fixed per-sequence state: GDN recurrent + GDN conv + PLE conv history. The geometry
    // accessors are constexpr and depend only on the target constants, so a default model view
    // suffices (no host artifact needed).
    const detail::ModelView geometry;
    const std::size_t per_sequence_state =
        geometry.gdn_state_bytes_per_sequence() +
        geometry.gdn_conv_state_bytes_per_sequence() +
        geometry.ple_conv_history_bytes_per_sequence();
    const std::size_t minimum_reservation =
        static_cast<std::size_t>(max_concurrency) * per_sequence_state;
    // Main KV + indexer side cache + MTP KV + MTP side cache per token, times the page-group
    // token count.
    const std::size_t stride =
        static_cast<std::size_t>(kMainPageTokens) *
        (geometry.main_kv_bytes_per_token() + geometry.side_cache_bytes_per_token() +
         geometry.mtp_kv_bytes_per_token() + geometry.mtp_side_cache_bytes_per_token());

    const runtime::SequenceCapacityCurve curve{
        .main_page_tokens                   = kMainPageTokens,
        .minimum_main_page_groups           = kMinimumPageGroups,
        .maximum_main_page_groups           = kMaximumPageGroups,
        .minimum_device_reservation_bytes   = minimum_reservation,
        .bytes_per_additional_main_page_group = stride,
    };
    return SequencePlanner(curve, options.max_context, max_concurrency);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return std::make_unique<detail::Program>(model.impl_->model, std::move(plan), device);
}

} // namespace ninfer::targets::qwen4exp
