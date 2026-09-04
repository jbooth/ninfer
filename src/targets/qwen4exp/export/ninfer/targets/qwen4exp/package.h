#pragma once

// qwen4exp (Qwen3.8-Flash-Next / jbq4) target package. Single-target, non-templated family.
// Reuses the qwen3_6 frontend; owns its own identity-free execution surface (see runtime.h).

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen4exp/runtime.h>
#include <ninfer/targets/qwen4exp/model_view.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen4exp {

struct Package;

namespace detail {

enum class WeightsProfile : std::uint8_t {
    Qwen4expJbQ4,
};

// The plan produced by the binder. Carries the materialization plan (device / host / validate-only
// split) and the binding metadata needed to build the ModelView (host object offsets + PLE
// descriptors).
class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

    // Binding metadata (visible to the package only).
    const HostArtifactView& host_view() const;
    const qwen3_6::FrontendResourcePlan& frontend_plan() const;

private:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen4exp::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;

    // Package-only accessors.
    const ModelView& view() const;

private:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen4exp::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-flash-next";
    static constexpr std::string_view weights_id = "jbq4";
    static constexpr std::string_view target_key = "qwen4exp";

    using WeightsProfile             = detail::WeightsProfile;
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = qwen3_6::Frontend;
    using PreparedPrompt             = qwen3_6::PreparedPrompt;
    using OutputSession              = qwen3_6::OutputSession;
    using PublishedOutput            = qwen3_6::PublishedOutput;
    using SequencePlanner            = SequencePlanner;
    using SequencePlan               = SequencePlan;
    using RequestBasePlan            = RequestBasePlan;
    using AdmissionCandidate         = AdmissionCandidate;
    using ResourcePlan               = ResourcePlan;
    using PersistentBackfillProof    = PersistentBackfillProof;
    using SequenceHandle             = SequenceHandle;
    using ContinuationHandle         = ContinuationHandle;
    using SharedPrefixHandle         = SharedPrefixHandle;
    using CaptureOffer               = CaptureOffer;
    using CacheSessionKey            = qwen3_6::PreparedSessionKey;
    using ContinuationSummary        = ContinuationSummary;
    using SharedPrefixSummary        = SharedPrefixSummary;
    using PressurePlanningSession    = PressurePlanningSession;
    using PressureTargetHandle       = PressureTargetHandle;
    using MaterializationResult      = MaterializationResult;
    using ContextTransactionProgress = ContextTransactionProgress;
    using CaptureAssessment          = CaptureAssessment;
    using ActiveCaptureResult        = ActiveCaptureResult;
    using PendingBatch               = PendingBatch;
    using StartResult                = StartResult;
    using PrefillProgress            = PrefillProgress;
    using CommitResult               = CommitResult;
    using DiscardResult              = DiscardResult;
    using FinishResult               = FinishResult;
    using AbortResult                = AbortResult;
    using ReleaseResult              = ReleaseResult;
    using Program                    = Program;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen4exp
} // namespace ninfer
