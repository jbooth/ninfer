#include <ninfer/targets/qwen4exp/runtime.h>
#include <ninfer/targets/qwen4exp/model_view.h>
#include "targets/qwen4exp/impl/config.h"
#include "targets/qwen4exp/impl/cpu/cpu.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <algorithm>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {
using namespace cfg;
using qwen3_6::PreparedPromptAccess;
using qwen3_6::PreparedPromptData;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument(message); }

void ensure(DeviceBuffer& buffer, std::size_t need) {
    if (buffer.bytes < need) { buffer = DeviceBuffer(need); }
}

} // namespace

// ---------------------------------------------------------------------------
// Program::Impl — the target's runtime state.  All GPU op leaves are stubs: the per-layer
// dataflow is a no-op that leaves the logits at zero.  The only real work is the host token
// embedding gather, the MoE expert streaming, and the GPU sampling of the (zero) logits.
// ---------------------------------------------------------------------------
struct Program::Impl {
    struct Lane {
        bool active          = false;
        bool prefill_done    = false;
        std::uint32_t generated = 0;
        std::vector<TokenId> tokens; // prompt + committed generated tokens (host)
    };

    const ModelView* model = nullptr;
    const DeviceContext* device = nullptr;
    SequencePlan plan;
    std::vector<Lane> lanes;

    bool transaction_active = false;
    runtime::LaneId transaction_lane{};

    DeviceBuffer workspace;      // activations (x) + logits for the current round.
    DeviceBuffer expert_gu;      // streamed gate_up slots (topk * slot).
    DeviceBuffer expert_dn;      // streamed down slots.
    DeviceBuffer sample_ws;      // sampling arena.
    DeviceBuffer sample_configs;
    DeviceBuffer sample_out;
    DeviceBuffer sample_pos;

    std::vector<TokenId> pending_tokens;        // stable storage for PendingBatch.tokens.
    std::vector<std::int32_t> pending_row_counts;

    std::uint64_t resource_revision = 1;

    Impl(const ModelView& model_, SequencePlan&& plan_, DeviceContext& device_)
        : model(&model_), device(&device_) {
        plan = std::move(plan_);
        const std::uint32_t maxc = plan.max_concurrency();
        lanes.resize(maxc);

        std::vector<ops::SamplingConfig> cfgs(maxc);
        for (auto& c : cfgs) {
            c.temperature       = 0.0f; // greedy
            c.top_k             = 0;
            c.top_p             = 1.0f;
            c.min_p             = 0.0f;
            c.presence_penalty  = 0.0f;
            c.frequency_penalty = 0.0f;
            c.seed              = 0;
            c.token_counts      = nullptr;
        }
        sample_configs = DeviceBuffer(static_cast<std::size_t>(maxc) * sizeof(ops::SamplingConfig));
        sample_configs.copy_from_host(cfgs.data(), sample_configs.bytes);
        sample_out = DeviceBuffer(static_cast<std::size_t>(maxc) * sizeof(std::int32_t));
        sample_pos = DeviceBuffer(static_cast<std::size_t>(maxc) * sizeof(std::int32_t));
        sample_ws = DeviceBuffer(ops::sampling_workspace_capacity_bytes(
            static_cast<std::int32_t>(vocab), 1, static_cast<std::int32_t>(maxc)));
    }

    // Runs the (stubbed) 48-layer dataflow for `input` (host token ids) and samples one token per
    // position.  The token embedding gather + MoE expert streaming are the real CPU components;
    // the fused layer dataflow is a no-op that leaves the logits at zero.
    std::vector<TokenId> run_forward(const std::span<const TokenId>& input) {
        const int B = static_cast<int>(input.size());
        const cudaStream_t stream = device->stream;
        const HostArtifactView& host = model->host;

        const std::size_t x_bytes      = static_cast<std::size_t>(hidden) * B * 2U;
        const std::size_t logits_bytes = static_cast<std::size_t>(vocab) * B * 2U;
        ensure(workspace, x_bytes + logits_bytes);
        auto* base       = static_cast<std::byte*>(workspace.p);
        auto* x_ptr      = base;
        auto* logits_ptr = base + x_bytes;

        // K0: gather the token embeddings (real CPU pread + H2D).
        gather_token_embedding(host, input.data(), B, x_ptr, stream);

        // MoE expert streaming (real CPU): deterministic selection of the first `topk` experts.
        const auto& bank = host.routed[0];
        const std::size_t gu_slot = bank.gate_up_bytes / static_cast<std::size_t>(moe_experts);
        const std::size_t dn_slot = bank.down_bytes / static_cast<std::size_t>(moe_experts);
        ensure(expert_gu, static_cast<std::size_t>(moe_topk) * gu_slot);
        ensure(expert_dn, static_cast<std::size_t>(moe_topk) * dn_slot);
        std::vector<std::uint8_t> selected(moe_topk);
        for (int k = 0; k < moe_topk; ++k) { selected[k] = static_cast<std::uint8_t>(k); }
        stream_experts(host, 0, input.data(), selected.data(), moe_topk,
                       static_cast<std::byte*>(expert_gu.p), static_cast<std::byte*>(expert_dn.p),
                       stream);

        // Stub: the fused 48-layer dataflow + output head leave the logits at zero.
        CUDA_CHECK(cudaMemsetAsync(logits_ptr, 0, logits_bytes, stream));

        // Sampling (real GPU kernel).
        std::vector<std::int32_t> pos(B);
        for (int i = 0; i < B; ++i) { pos[i] = i; }
        CUDA_CHECK(cudaMemcpyAsync(sample_pos.p, pos.data(), static_cast<std::size_t>(B) * 4,
                                   cudaMemcpyHostToDevice, stream));
        WorkspaceArena ws(DeviceSpan{sample_ws.p, sample_ws.bytes});
        Tensor logits(logits_ptr, DType::BF16, {static_cast<std::int32_t>(vocab), B});
        Tensor out(sample_out.p, DType::I32, {B});
        Tensor positions(sample_pos.p, DType::I32, {B});
        ops::sample(logits, out, static_cast<std::int32_t>(vocab),
                    static_cast<const ops::SamplingConfig*>(sample_configs.p), positions,
                    B > 1 ? ops::kSamplePurposeDecode : ops::kSamplePurposePrefill, ws, stream);

        std::vector<TokenId> tokens(B);
        CUDA_CHECK(cudaMemcpyAsync(tokens.data(), sample_out.p, static_cast<std::size_t>(B) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return tokens;
    }

    PendingBatch make_batch(const void* owner, const std::span<const SequenceHandle>& rows,
                            const std::vector<TokenId>& tokens) {
        const std::size_t B = rows.size();
        pending_tokens.assign(tokens.begin(), tokens.end());
        pending_row_counts.assign(B, 1);
        return PendingBatch::make_pending(
            owner, 1, rows, std::span<const TokenId>{pending_tokens.data(), B},
            std::span<const std::int32_t>{pending_row_counts.data(), B}, 1,
            runtime::ExecutionTiming{});
    }
};

// ---------------------------------------------------------------------------
// Public Program members.
// ---------------------------------------------------------------------------
Program::Program(const ModelView& model, SequencePlan&& plan, DeviceContext& device)
    : impl_(std::make_unique<Impl>(model, std::move(plan), device)) {}
Program::~Program() = default;

bool Program::has_context_transaction() const noexcept { return impl_->transaction_active; }

RequestBasePlan
Program::plan_request(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options) const {
    const auto& data = PreparedPromptAccess::view(prompt);
    runtime::RequestPlanSummary summary;
    summary.prompt_tokens           = static_cast<std::uint32_t>(data.token_ids.size());
    summary.reusable_prompt_tokens  = 0;
    summary.requested_output_tokens = options.requested_output_tokens;
    summary.effective_output_tokens = options.requested_output_tokens;
    summary.effective_limit_reason  = FinishReason::OutputLimit;
    summary.prefix_reuse_path       = PrefixReusePath::Root;
    summary.service_work_quanta =
        static_cast<std::uint64_t>(summary.prompt_tokens) + summary.effective_output_tokens;
    summary.publish_continuation    = false;
    return RequestBasePlan(std::move(summary), data.context_cache, std::nullopt);
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const PreparedPrompt& prompt, const RequestBasePlan& base,
                           runtime::LaneId lane, const ContinuationHandle* source,
                           const SharedPrefixHandle* shared_source,
                           std::optional<runtime::CheckpointRef> checkpoint,
                           bool publish_continuation,
                           const runtime::ContextMachineCostModel& machine) const {
    (void)prompt; (void)source; (void)shared_source; (void)checkpoint;
    (void)publish_continuation; (void)machine;
    runtime::IdentityMaterializationAssessment identity;
    identity.physical_status    = runtime::MaterializationPhysicalStatus::Feasible;
    identity.source_disposition = runtime::ClaimDisposition::ConsumedToActive;
    identity.expandable         = false;
    identity.projection_work    = 0;
    identity.assessment_digest  = 0;
    return AdmissionCandidate(base.summary(), std::move(identity), lane);
}

std::optional<ResourcePlan> Program::seal_identity(const AdmissionCandidate& candidate,
                                                   const PreparedPrompt& prompt) {
    (void)prompt;
    return ResourcePlan(candidate.summary(), false, impl_->resource_revision, candidate.lane());
}

PressurePlanningSession
Program::begin_pressure_planning(const runtime::ContextMachineCostModel& machine,
                                 std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const ContinuationHandle* const> continuations,
                                 std::span<const std::uint32_t> continuation_ordinals,
                                 std::span<const SharedPrefixHandle* const> shared_prefixes,
                                 std::span<const std::uint32_t> shared_ordinals) {
    return PressurePlanningSession(machine, candidates, continuations, continuation_ordinals,
                                   shared_prefixes, shared_ordinals);
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    (void)cancellation;
    if (plan.resource_revision() == 0 ||
        plan.resource_revision() != impl_->resource_revision) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    PreparedPromptData data = PreparedPromptAccess::take(std::move(prompt));
    auto& lane_state = impl_->lanes[plan.lane().value];
    lane_state.active       = true;
    lane_state.prefill_done = false;
    lane_state.generated    = 0;
    lane_state.tokens       = std::move(data.token_ids);
    impl_->transaction_active = true;
    impl_->transaction_lane   = plan.lane();
    return runtime::ContextTransactionReserveStatus::Reserved;
}

ContextTransactionProgress Program::progress_context_transaction(
    runtime::CancellationFlagView cancellation) {
    (void)cancellation;
    if (!impl_->transaction_active) {
        return ContextTransactionProgress{runtime::ContextTransactionInProgress{}};
    }
    const runtime::LaneId lane = impl_->transaction_lane;
    MaterializationResult out;
    out.status    = runtime::ContextTransactionStatus::Published;
    out.published = StartResult{SequenceHandle(static_cast<const void*>(this), lane, 1)};
    return ContextTransactionProgress{std::move(out)};
}

void Program::finalize_context_transaction() noexcept {
    // The resource manager calls this from adopt() after the logical record is committed; the
    // program-side transaction ownership is released only there so the adopt path still observes
    // has_context_transaction() == true.
    impl_->transaction_active = false;
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& base, const ResourcePlan& plan,
                                   std::span<const SequenceHandle> active) const {
    (void)base; (void)plan; (void)active;
    return std::nullopt;
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence,
                                         runtime::ExecutionTiming* timing) {
    if (timing) { *timing = runtime::ExecutionTiming{}; }
    auto& lane_state = impl_->lanes[sequence.lane().value];
    if (lane_state.tokens.empty()) { fail("advance_prefill: lane has no prompt tokens"); }
    const std::size_t T = lane_state.tokens.size();
    std::span<const TokenId> last{lane_state.tokens.data() + T - 1, 1};
    std::vector<TokenId> sampled = impl_->run_forward(last);
    lane_state.prefill_done = true;
    std::vector<SequenceHandle> rows{sequence};
    PrefillProgress out;
    out.processed_prompt_tokens = static_cast<std::uint32_t>(T);
    out.complete                = true;
    out.summary.prompt_tokens   = static_cast<std::uint32_t>(T);
    out.pending = impl_->make_batch(static_cast<const void*>(this),
                                    std::span<const SequenceHandle>{rows.data(), 1}, sampled);
    return out;
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* private_source,
                         const SharedPrefixHandle* shared_source,
                         std::optional<runtime::CheckpointRef> checkpoint) const {
    (void)offer; (void)private_source; (void)shared_source; (void)checkpoint;
    return CaptureAssessment{};
}

bool Program::shared_capture_matches(const CaptureOffer& offer,
                                     const SharedPrefixHandle& shared) const {
    (void)offer; (void)shared;
    return false;
}

void Program::skip_capture(CaptureOffer&& offer) { (void)offer; }

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer, const SharedPrefixHandle* private_source,
                                const SharedPrefixHandle* shared_source,
                                std::optional<runtime::CheckpointRef> checkpoint,
                                runtime::CancellationFlagView cancellation) {
    (void)offer; (void)private_source; (void)shared_source; (void)checkpoint;
    (void)cancellation;
    return runtime::ContextTransactionReserveStatus::Aborted;
}

PendingBatch Program::decode(std::span<const SequenceHandle> rows,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* timing) {
    (void)budgets;
    if (timing) { *timing = runtime::ExecutionTiming{}; }
    const std::size_t B = rows.size();
    std::vector<TokenId> input(B);
    std::vector<SequenceHandle> rows_copy(B);
    for (std::size_t i = 0; i < B; ++i) {
        auto& lane_state = impl_->lanes[rows[i].lane().value];
        if (lane_state.tokens.empty()) { fail("decode: lane has no tokens"); }
        input[i]     = lane_state.tokens.back();
        rows_copy[i] = rows[i];
    }
    std::vector<TokenId> sampled = impl_->run_forward({input.data(), B});
    return impl_->make_batch(static_cast<const void*>(this),
                             std::span<const SequenceHandle>{rows_copy.data(), B}, sampled);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> rows,
                              std::span<const TokenId> tokens, std::uint32_t row_stride,
                              runtime::ExecutionTiming* timing) {
    if (timing) { *timing = runtime::ExecutionTiming{}; }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto& lane_state = impl_->lanes[rows[i].lane().value];
        for (std::size_t k = 0; k < row_stride; ++k) {
            lane_state.tokens.push_back(tokens[i * row_stride + k]);
            lane_state.generated++;
        }
    }
    return runtime::ExecutionTiming{};
}

CommitResult Program::commit(PendingBatch&& batch,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation observation,
                             runtime::ExecutionTiming* timing) {
    (void)observation;
    if (timing) { *timing = runtime::ExecutionTiming{}; }
    CommitResult out;
    out.row_count = batch.row_count();
    for (std::size_t i = 0; i < batch.row_count(); ++i) {
        const runtime::LaneId lane = batch.rows()[i].lane();
        auto& lane_state  = impl_->lanes[lane.value];
        lane_state.tokens.push_back(batch.tokens()[i]);
        lane_state.generated++;
        // The engine derives the expected disposition from the same decision: terminal rows become
        // finishable, live rows stay active. Cancelled rows are handled by the engine before commit.
        out.rows[i].disposition = decisions[i].terminal
                                      ? runtime::CommitDisposition::Finishable
                                      : runtime::CommitDisposition::Active;
    }
    return out;
}

DiscardResult Program::abort_pending(PendingBatch&& batch) noexcept {
    return DiscardResult{runtime::ConsumeStatus::Consumed, batch.row_count()};
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    impl_->lanes[sequence.lane().value].active = false;
    FinishResult out;
    out.status      = runtime::ConsumeStatus::Consumed;
    out.disposition = runtime::FinishDisposition::Released;
    return out;
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    impl_->lanes[sequence.lane().value].active = false;
    AbortResult out;
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

ReleaseResult Program::release_continuation(ContinuationHandle&& handle) noexcept {
    (void)handle;
    return ReleaseResult{runtime::ConsumeStatus::Consumed};
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& handle) noexcept {
    (void)handle;
    return ReleaseResult{runtime::ConsumeStatus::Consumed};
}

void Program::fail_all_cleanup() noexcept {}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    (void)base;
    return true;
}

std::uint64_t Program::resource_revision() const noexcept { return impl_->resource_revision; }

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    PhysicalUsageSnapshot out;
    out.resource_revision = impl_->resource_revision;
    return out;
}

MemorySummary Program::memory_summary() const noexcept {
    MemorySummary out;
    out.device                  = impl_->device->device;
    out.max_context             = impl_->plan.capacity();
    out.kv_capacity             = impl_->plan.kv_capacity();
    out.runtime_reservation_bytes = impl_->plan.device_reservation_bytes();
    out.workspace_logical_peak_bytes =
        impl_->workspace.bytes + impl_->sample_ws.bytes + impl_->expert_gu.bytes +
        impl_->expert_dn.bytes;
    return out;
}

void Program::reset_memory_peaks() noexcept {
    // The skeleton reuses fixed scratch buffers; there are no peaks to reset.
}

// ---------------------------------------------------------------------------
// Pressure planning is never entered by the degenerate (no-cache) skeleton; the fast-path planner
// seals identity directly.  The session and its methods exist only to satisfy the Program
// contract, so the unsupported operations throw defensively.
// ---------------------------------------------------------------------------
PressurePlanningSession::PressurePlanningSession(
    const runtime::ContextMachineCostModel& cost,
    std::span<const AdmissionCandidate* const> candidates,
    std::span<const ContinuationHandle* const> continuations,
    std::span<const std::uint32_t> continuation_ordinals,
    std::span<const SharedPrefixHandle* const> shared_prefixes,
    std::span<const std::uint32_t> shared_ordinals)
    : generation_(1) {
    (void)cost; (void)continuations; (void)continuation_ordinals;
    (void)shared_prefixes; (void)shared_ordinals;
    slots_.reserve(candidates.size());
    for (const auto* candidate : candidates) {
        Slot slot;
        slot.candidate = candidate;
        slots_.push_back(slot);
    }
}

PressureTargetHandle PressurePlanningSession::identity_target(const AdmissionCandidate&) const {
    throw std::logic_error("qwen4exp skeleton does not support pressure planning");
}

PressureTargetHandle PressurePlanningSession::root_maximal_target(const AdmissionCandidate&) {
    throw std::logic_error("qwen4exp skeleton does not support pressure planning");
}

runtime::PressureTargetAssessment PressurePlanningSession::assess(PressureTargetHandle) {
    throw std::logic_error("qwen4exp skeleton does not support pressure planning");
}

PreparedPressureExpansion PressurePlanningSession::prepare_expansion(PressureTargetHandle) {
    throw std::logic_error("qwen4exp skeleton does not support pressure planning");
}

PressureExpansionView PressurePlanningSession::commit_expansion(PreparedPressureExpansion&&) {
    throw std::logic_error("qwen4exp skeleton does not support pressure planning");
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&&) noexcept {}

std::optional<ResourcePlan> PressurePlanningSession::seal(PressureTargetHandle,
                                                          const PreparedPrompt&) {
    return std::nullopt;
}

} // namespace ninfer::targets::qwen4exp::detail

namespace ninfer::targets::qwen4exp {

// Build the concrete sequence plan from the resolved KV capacity. The degenerate target keeps a
// fixed device reservation (per-sequence state) plus the affine Main-KV reservation, and a fixed
// workspace budget for the per-round scratch.
SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (curve_.main_page_tokens == 0) { throw std::logic_error("sequence planner has no curve"); }
    const std::uint32_t kv_capacity = curve_.resolved_tokens(main_page_groups);
    const std::uint32_t capacity =
        max_context_ == 0 ? kv_capacity : std::min(max_context_, kv_capacity);
    const std::size_t device_reservation = curve_.reservation_bytes(main_page_groups);
    constexpr std::size_t kWorkspaceCapacityBytes = 256ULL * 1024 * 1024;
    return SequencePlan(capacity, kv_capacity, max_concurrency_, device_reservation,
                        kWorkspaceCapacityBytes);
}

} // namespace ninfer::targets::qwen4exp
