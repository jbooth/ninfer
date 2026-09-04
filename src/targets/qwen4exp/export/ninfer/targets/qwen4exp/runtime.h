#pragma once

// qwen4exp (Qwen3.8-Flash-Next / jbq4) family runtime contract.
//
// Phase-2 skeleton: a single-target, non-templated family. It reuses the qwen3_6 frontend
// (PreparedPrompt / OutputSession / PublishedOutput / Frontend) and the shared prepared-prompt
// cache types, and defines its own identity-free execution surface. The Program runs a fully
// eager, text-only, no-CUDA-graph dataflow: real binder-backed weight views, real CPU gather
// (token/PLE) and MoE expert streaming, real shape-generic kernels where registered, and
// deterministic constant-fill op leaves elsewhere (garbage output by design).

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::runtime {
struct ContextMachineCostModel;
}

namespace ninfer::targets::qwen4exp {

// Frontend reuse: the artifact embeds the same six frontend resources as the qwen3.6 family.
using qwen3_6::Frontend;
using qwen3_6::FrontendOptions;
using qwen3_6::PreparedPrompt;
using qwen3_6::OutputSession;
using qwen3_6::PublishedOutput;
using qwen3_6::PreparedContextCache;
using qwen3_6::PreparedSessionKey;
using qwen3_6::PreparedPromptAccess;

namespace detail {
struct ModelView;
class Program;
class PressurePlanningSession;
} // namespace detail

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

struct PhysicalUsageSnapshot {
    std::uint64_t resource_revision = 0;
    std::uint32_t device_state_slots = 0;
    std::uint32_t host_state_slots   = 0;
    std::uint32_t device_main_kv_pages = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes        = 0;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalUsageSnapshot&,
                                                   const PhysicalUsageSnapshot&) noexcept = default;
};

// Program-minted shortlist metadata. Unused by the no-cache qwen4exp target but kept so the
// type surface matches the shared contract used by capture/continuation summaries.
struct PrefixShortlistKey {
    std::uint64_t digest       = 0;
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefixShortlistKey,
                                                   PrefixShortlistKey) noexcept = default;
};

struct TargetKVRequirement {
    std::uint32_t main_frontier    = 0;
    std::uint32_t backend_frontier = 0;
    std::uint32_t main_pages       = 0;
    std::uint32_t backend_pages    = 0;

    [[nodiscard]] friend constexpr bool operator==(TargetKVRequirement,
                                                   TargetKVRequirement) noexcept = default;
};

struct CheckpointSummary {
    runtime::CheckpointRef ref;
    runtime::CheckpointScope scope = runtime::CheckpointScope::Private;
    PrefixShortlistKey shortlist_key;
    runtime::ReplicaResidency state_residency = runtime::ReplicaResidency::DeviceOnly;
    TargetKVRequirement required_kv;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend bool operator==(const CheckpointSummary&, const CheckpointSummary&)
        noexcept = default;
};

struct ContinuationSummary {
    std::optional<CheckpointSummary> endpoint;
    std::optional<CheckpointSummary> rewrite;
    std::vector<CheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;
};

struct SharedPrefixSummary {
    CheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    [[nodiscard]] friend bool operator==(const SharedPrefixSummary&, const SharedPrefixSummary&)
        noexcept = default;
};

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

class SequenceHandle {
public:
    SequenceHandle() noexcept = default;
    ~SequenceHandle()         = default;

    SequenceHandle(const SequenceHandle&)            = default;
    SequenceHandle& operator=(const SequenceHandle&) = default;
    SequenceHandle(SequenceHandle&&) noexcept        = default;
    SequenceHandle& operator=(SequenceHandle&&) noexcept = default;

    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }

private:
    SequenceHandle(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch) {}

    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend detail::Program;
};

// Degenerate in the no-cache, no-continuation qwen4exp target. Never minted; kept for the
// shared contract surface.
class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&&) noexcept            = default;
    ContinuationHandle& operator=(ContinuationHandle&&) noexcept = default;
    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

    [[nodiscard]] bool valid() const noexcept { return index_ != 0; }

private:
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend detail::Program;
};

class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&&) noexcept            = default;
    SharedPrefixHandle& operator=(SharedPrefixHandle&&) noexcept = default;
    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

    [[nodiscard]] bool valid() const noexcept { return index_ != 0; }

private:
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;

    friend detail::Program;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&&) noexcept            = default;
    CaptureOffer& operator=(CaptureOffer&&) noexcept = default;
    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;

    friend detail::Program;
};

class PressureTargetHandle {
public:
    PressureTargetHandle() noexcept = default;

    [[nodiscard]] friend constexpr bool operator==(PressureTargetHandle,
                                                   PressureTargetHandle) noexcept = default;

private:
    const void* session_      = nullptr;
    std::uint32_t generation_ = 0;
    std::uint32_t index_      = 0;

    friend detail::Program;
    friend detail::PressurePlanningSession;
};

// Never minted by the no-cache qwen4exp target; prove_persistent_backfill returns nullopt.
class PersistentBackfillProof {
public:
    PersistentBackfillProof() noexcept = default;
    ~PersistentBackfillProof()         = default;

    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;
    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    explicit PersistentBackfillProof(std::uint64_t revision) noexcept : revision_(revision) {}

    std::uint64_t revision_ = 0;

    friend class detail::Program;
};

// ---------------------------------------------------------------------------
// Data containers
// ---------------------------------------------------------------------------

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&&) noexcept            = default;
    PendingBatch& operator=(PendingBatch&&) noexcept = default;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }

    [[nodiscard]] std::span<const SequenceHandle> rows() const noexcept {
        return {rows_.data(), row_count_};
    }

    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }

    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }

    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }

    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

    // Package/Program-only factory: `tokens` and `row_counts` must reference caller-stable storage
    // (the owning Program) that outlives the batch.
    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction, std::span<const SequenceHandle> rows,
                 std::span<const TokenId> tokens, std::span<const std::int32_t> row_counts,
                 std::uint32_t row_stride, runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_{};

    friend detail::Program;
};

struct PrefillProgress {
    runtime::BeginSummary summary{};
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing{};
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    bool pressure_committed               = false;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedVictimResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    bool pressure_committed               = false;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<ContinuationSummary> final_summary;
};

struct MaterializationSharedSourceResult {
    runtime::ClaimDisposition disposition = runtime::ClaimDisposition::Retained;
    std::optional<SharedPrefixSummary> final_summary;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<MaterializationSourceResult> source;
    std::optional<MaterializationSharedSourceResult> shared_source;
    std::vector<MaterializationVictimResult> victims;
    std::vector<MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations{};
};

struct CaptureAssessment {
    PrefixShortlistKey shortlist_key{};
    runtime::PrefillWork protected_rebuild_work{};
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::vector<runtime::CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier = 0;
    bool publishes_private = false;
    bool publishes_shared  = false;
    bool needs_transfer    = false;
    bool recycles_private_state = false;
};

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations{};
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult,
                 ActiveCaptureResult>;

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings{};
    SpeculativeStats speculative{};
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    // Prompt-frontier captures become valid only after the generated Begin token is committed.
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing{};
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings{};
    SpeculativeStats speculative{};
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings{};
    SpeculativeStats speculative{};
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

// ---------------------------------------------------------------------------
// Planner / plan types
// ---------------------------------------------------------------------------

class SequencePlan {
public:
    SequencePlan() noexcept                    = default;
    ~SequencePlan()                            = default;

    SequencePlan(SequencePlan&&) noexcept            = default;
    SequencePlan& operator=(SequencePlan&&) noexcept = default;
    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::uint32_t kv_capacity() const noexcept { return kv_capacity_; }

    [[nodiscard]] std::uint32_t max_concurrency() const noexcept { return max_concurrency_; }

    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept {
        return device_reservation_bytes_;
    }

    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept {
        return workspace_capacity_bytes_;
    }

private:
    SequencePlan(std::uint32_t capacity, std::uint32_t kv_capacity, std::uint32_t max_concurrency,
                 std::size_t device_reservation_bytes, std::size_t workspace_capacity_bytes) noexcept
        : capacity_(capacity), kv_capacity_(kv_capacity), max_concurrency_(max_concurrency),
          device_reservation_bytes_(device_reservation_bytes),
          workspace_capacity_bytes_(workspace_capacity_bytes) {}

    std::uint32_t capacity_             = 0;
    std::uint32_t kv_capacity_          = 0;
    std::uint32_t max_concurrency_      = 0;
    std::size_t device_reservation_bytes_ = 0;
    std::size_t workspace_capacity_bytes_ = 0;

    friend detail::Program;
    friend class SequencePlanner;
};

class SequencePlanner {
public:
    SequencePlanner() noexcept = default;
    ~SequencePlanner()         = default;

    SequencePlanner(SequencePlanner&&) noexcept            = default;
    SequencePlanner& operator=(SequencePlanner&&) noexcept = default;
    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept {
        return curve_;
    }

    SequencePlan finalize(std::uint32_t main_page_groups) &&;

private:
    SequencePlanner(runtime::SequenceCapacityCurve curve, std::uint32_t max_context,
                    std::uint32_t max_concurrency) noexcept
        : curve_(std::move(curve)), max_context_(max_context), max_concurrency_(max_concurrency) {}

    runtime::SequenceCapacityCurve curve_{};
    std::uint32_t max_context_     = 0;
    std::uint32_t max_concurrency_ = 1;

    friend class Package;
    friend detail::Program;
};

class RequestBasePlan {
public:
    RequestBasePlan() noexcept = default;
    ~RequestBasePlan()         = default;

    RequestBasePlan(RequestBasePlan&&) noexcept            = default;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept = default;
    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return summary_; }

    [[nodiscard]] const PreparedContextCache& context_cache() const noexcept {
        return context_cache_;
    }

    [[nodiscard]] std::optional<PrefixShortlistKey> prefix_shortlist_key(
        std::uint32_t) const noexcept {
        return shortlist_key_;
    }

private:
    RequestBasePlan(runtime::RequestPlanSummary summary, PreparedContextCache context_cache,
                    std::optional<PrefixShortlistKey> shortlist_key) noexcept
        : summary_(std::move(summary)), context_cache_(std::move(context_cache)),
          shortlist_key_(std::move(shortlist_key)) {}

    runtime::RequestPlanSummary summary_{};
    PreparedContextCache context_cache_{};
    std::optional<PrefixShortlistKey> shortlist_key_{};

    friend detail::Program;
};

class AdmissionCandidate {
public:
    AdmissionCandidate() noexcept = default;
    ~AdmissionCandidate()         = default;

    AdmissionCandidate(AdmissionCandidate&&) noexcept            = default;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept = default;
    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return summary_; }

    [[nodiscard]] const runtime::IdentityMaterializationAssessment& identity_assessment()
        const noexcept {
        return identity_;
    }

    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }

private:
    AdmissionCandidate(runtime::RequestPlanSummary summary,
                       runtime::IdentityMaterializationAssessment identity,
                       runtime::LaneId lane) noexcept
        : summary_(std::move(summary)), identity_(std::move(identity)), lane_(lane) {}

    runtime::RequestPlanSummary summary_{};
    runtime::IdentityMaterializationAssessment identity_{};
    runtime::LaneId lane_{};

    friend detail::Program;
};

class ResourcePlan {
public:
    ResourcePlan() noexcept = default;
    ~ResourcePlan()         = default;

    ResourcePlan(ResourcePlan&&) noexcept            = default;
    ResourcePlan& operator=(ResourcePlan&&) noexcept = default;
    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return summary_; }

    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }

private:
    ResourcePlan(runtime::RequestPlanSummary summary, bool needs_transfer,
                 std::uint64_t revision, runtime::LaneId lane) noexcept
        : summary_(std::move(summary)), needs_transfer_(needs_transfer), revision_(revision),
          lane_(lane) {}

    runtime::RequestPlanSummary summary_{};
    bool needs_transfer_ = false;
    std::uint64_t revision_ = 0;
    runtime::LaneId lane_{};

    friend detail::Program;
    friend detail::PressurePlanningSession;
};

// ---------------------------------------------------------------------------
// Pressure planning session (degenerate: no cache / no continuation / no expansion)
// ---------------------------------------------------------------------------

class PreparedPressureExpansion {
public:
    PreparedPressureExpansion() noexcept                         = default;
    PreparedPressureExpansion(PreparedPressureExpansion&&) noexcept = default;
    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&) noexcept = default;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept { return new_canonical_count_; }

private:
    PreparedPressureExpansion(const void* session, std::uint32_t generation,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), generation_(generation), new_canonical_count_(new_canonical_count) {}

    const void* session_             = nullptr;
    std::uint32_t generation_        = 0;
    std::uint32_t new_canonical_count_ = 0;

    friend detail::PressurePlanningSession;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children{};
    std::uint32_t new_canonical_count = 0;
};

namespace detail {
class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept            = default;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept = default;
    ~PressurePlanningSession()                                             = default;

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(const AdmissionCandidate& candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(const AdmissionCandidate& root_candidate);
    [[nodiscard]] runtime::PressureTargetAssessment assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    [[nodiscard]] PressureExpansionView
    commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&&) noexcept;
    [[nodiscard]] std::optional<ResourcePlan>
    seal(PressureTargetHandle target, const PreparedPrompt& prompt);

    // Minted by Program::begin_pressure_planning.
    PressurePlanningSession(const runtime::ContextMachineCostModel& cost,
                            std::span<const AdmissionCandidate* const> candidates,
                            std::span<const ContinuationHandle* const> continuations,
                            std::span<const std::uint32_t> continuation_ordinals,
                            std::span<const SharedPrefixHandle* const> shared_prefixes,
                            std::span<const std::uint32_t> shared_ordinals);

private:
    struct Slot {
        const AdmissionCandidate* candidate = nullptr;
        PressureTargetHandle handle{};
    };
    std::uint64_t generation_ = 0;
    std::vector<Slot> slots_{};
    std::vector<PressureTargetHandle> assessed_handles_{};

    friend class Program;
};
} // namespace detail

// Public (engine-facing) pressure session type alias to the detail implementation.
using PressurePlanningSession = detail::PressurePlanningSession;

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

namespace detail {
class Program {
public:
    Program(const ModelView& model, SequencePlan&& plan, DeviceContext& device);
    ~Program();

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // -- planning / admission --
    [[nodiscard]] RequestBasePlan plan_request(
        const PreparedPrompt& prompt, const runtime::ResolvedExecutionOptions& options) const;
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base, runtime::LaneId lane,
                      const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
                      std::optional<runtime::CheckpointRef> checkpoint,
                      bool publish_continuation,
                      const runtime::ContextMachineCostModel& machine) const;
    [[nodiscard]] std::optional<ResourcePlan>
    seal_identity(const AdmissionCandidate& candidate, const PreparedPrompt& prompt);
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(const runtime::ContextMachineCostModel& machine,
                            std::span<const AdmissionCandidate* const> candidates,
                            std::span<const ContinuationHandle* const> continuations,
                            std::span<const std::uint32_t> continuation_ordinals,
                            std::span<const SharedPrefixHandle* const> shared_prefixes,
                            std::span<const std::uint32_t> shared_ordinals);

    // -- resource transaction (degenerate: immediate publish, no source / no victims) --
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;

    // -- backfill / capture (degenerate: none) --
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& base, const ResourcePlan& plan,
                              std::span<const SequenceHandle> active) const;
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* private_source,
                    const SharedPrefixHandle* shared_source,
                    std::optional<runtime::CheckpointRef> checkpoint) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* private_source,
                           const SharedPrefixHandle* shared_source,
                           std::optional<runtime::CheckpointRef> checkpoint,
                           runtime::CancellationFlagView cancellation);

    // -- execution --
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] PendingBatch
    decode(std::span<const SequenceHandle> rows, std::span<const runtime::RoundBudget> budgets,
           runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                         std::uint32_t row_stride, runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& batch, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& batch) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult release_continuation(ContinuationHandle&& handle) noexcept;
    [[nodiscard]] ReleaseResult release_shared_prefix(SharedPrefixHandle&& handle) noexcept;
    void fail_all_cleanup() noexcept;

    // -- invariants / usage --
    [[nodiscard]] bool isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] std::uint64_t resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace detail

using Program = detail::Program;

} // namespace ninfer::targets::qwen4exp
