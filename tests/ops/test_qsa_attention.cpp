// QSA sparse grouped-query attention (qwen4exp) Op qualification.
//
// The oracle is the shared naive FP64 attention oracle (tests/ops/softmax_attention/oracle.h)
// with a gather-restricted visible set: for token t, the live keys are the non-negative topk ids
// restricted to the causal prefix (j <= positions[t]) and the live extent (j < n_kv). Inputs are
// rounded to the BF16 grid first so the oracle reads exactly the represented values.

#include "ninfer/ops/qsa.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "softmax_attention/oracle.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKVHeads    = 2;
constexpr std::int32_t kTopK       = 2051;
constexpr float kScale             = 1.0F / 16.0F;

// The BF16 output's dominant error term is the reference's own storage rounding. The
// aggregate relative-L2 sits at the attention-family BF16 level; the gross pointwise cap must
// cover the worst-case BF16 half-ulp at a binade bottom (3.906e-3 relative) with margin.
constexpr ReductionCriterion kQsaBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-5,
    /*gross_relative_to_max_reference*/ 4.2e-3,
};

struct Case {
    std::int32_t n_kv;
    std::int32_t tokens;
    std::uint32_t seed;
};

// Per (token, key) visibility: key j is live for token t iff j is a non-negative topk id for t,
// j < n_kv, and j <= positions[t].
std::vector<bool> build_visible(const std::vector<std::int32_t>& topk, std::int32_t n_kv,
                                std::int32_t tokens, const std::vector<std::int32_t>& positions) {
    std::vector<bool> vis(static_cast<std::size_t>(n_kv) * tokens, false);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t pos = positions[t];
        for (std::int32_t slot = 0; slot < kTopK; ++slot) {
            const std::int32_t j = topk[static_cast<std::size_t>(slot) * tokens + t];
            if (j >= 0 && j < n_kv && j <= pos) vis[static_cast<std::size_t>(j) * tokens + t] = true;
        }
    }
    return vis;
}

std::int32_t run_case(const Case& c, std::mt19937& rng) {
    const std::int32_t n_kv  = c.n_kv;
    const std::int32_t tokens = c.tokens;

    // Represented BF16 inputs. q/out are [256, 24, T] dim-major: (d, h, t) at d*24*T + h*T + t.
    std::vector<float> q(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    std::vector<float> k(static_cast<std::size_t>(n_kv) * kKVHeads * kHeadDim);
    std::vector<float> v(static_cast<std::size_t>(n_kv) * kKVHeads * kHeadDim);
    fill_uniform(q, c.seed, -1.0F, 1.0F);
    fill_uniform(k, c.seed + 1U, -1.0F, 1.0F);
    fill_uniform(v, c.seed + 2U, -1.0F, 1.0F);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    // Causal prefix: positions[t] = (n_kv - tokens) + t, so the last query sees all keys.
    std::vector<std::int32_t> positions(tokens);
    for (std::int32_t t = 0; t < tokens; ++t) positions[t] = (n_kv - tokens) + t;

    // Build a per-token topk: a random live subset of the causal prefix plus a couple of
    // future (masked) ids to exercise the causal filter.
    std::vector<std::int32_t> topk(static_cast<std::size_t>(kTopK) * tokens, -1);
    std::uniform_int_distribution<std::int32_t> pick(0, kTopK - 1);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int32_t pos = positions[t];
        // One token is intentionally empty (all -1) to exercise the exact-zero path.
        if (t == 0) continue;
        const std::int32_t n_live =
            std::min<std::int32_t>(pos + 1, std::uniform_int_distribution<std::int32_t>(3, 40)(rng));
        std::vector<bool> used(static_cast<std::size_t>(n_kv), false);
        std::int32_t slot = 0;
        for (std::int32_t r = 0; r < n_live && slot < kTopK; ++r) {
            std::int32_t j;
            do { j = std::uniform_int_distribution<std::int32_t>(0, pos)(rng); } while (used[j]);
            used[j] = true;
            topk[static_cast<std::size_t>(slot++) * tokens + t] = j;
        }
        // Add up to two future ids (pos+1 .. n_kv-1) that must be masked by the causal prefix.
        for (std::int32_t j = pos + 1; j < n_kv && slot < kTopK && (j - pos) <= 2; ++j) {
            topk[static_cast<std::size_t>(slot++) * tokens + t] = j;
        }
    }

    // Independent FP64 oracle.
    const std::vector<bool> vis = build_visible(topk, n_kv, tokens, positions);
    std::vector<double> reference(static_cast<std::size_t>(kQueryHeads) * kHeadDim * tokens);
    auto query_value = [&](std::int32_t d, std::int32_t h, std::int32_t t) {
        return double(q[static_cast<std::size_t>(d) * kQueryHeads * tokens +
                        static_cast<std::size_t>(h) * tokens + t]);
    };
    auto key_value = [&](std::int32_t d, std::int32_t kvh, std::int32_t j) {
        return double(k[static_cast<std::size_t>(j) * kKVHeads * kHeadDim +
                        static_cast<std::size_t>(kvh) * kHeadDim + d]);
    };
    auto value_value = [&](std::int32_t d, std::int32_t kvh, std::int32_t j) {
        return double(v[static_cast<std::size_t>(j) * kKVHeads * kHeadDim +
                        static_cast<std::size_t>(kvh) * kHeadDim + d]);
    };
    auto visible = [&](std::int32_t t, std::int32_t j) { return vis[static_cast<std::size_t>(j) * tokens + t]; };
    auto store = [&](std::int32_t d, std::int32_t h, std::int32_t t, double val) {
        reference[static_cast<std::size_t>(d) * kQueryHeads * tokens + static_cast<std::size_t>(h) * tokens + t] = val;
    };
    naive_dense_softmax_attention(ninfer::ops::AttentionHeadGeometry{kHeadDim, kQueryHeads, kKVHeads}, tokens,
                                  n_kv, double(kScale), query_value, key_value, value_value, visible,
                                  store);

    // Device inputs + run.
    const DeviceBuffer dq = to_device_bf16(q), dk = to_device_bf16(k), dv = to_device_bf16(v),
        dtopk = to_device_i32(topk), dpos = to_device_i32(positions),
        dout(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * 2);
    const Tensor tq(dq.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    const Tensor tk(dk.p, DType::BF16, {n_kv, kKVHeads, kHeadDim});
    const Tensor tv(dv.p, DType::BF16, {n_kv, kKVHeads, kHeadDim});
    const Tensor ttok(dtopk.p, DType::I32, {kTopK, tokens});
    const Tensor tpos(dpos.p, DType::I32, {tokens});
    Tensor tout(dout.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    WorkspaceArena arena(4096);
    ops::qsa_softmax_attention(tq, ttok, tpos, tk, tv, n_kv,
                               ninfer::ops::AttentionHeadGeometry{kHeadDim, kQueryHeads, kKVHeads}, kScale,
                               arena, tout, 0);
    cuda_synchronize();

    const std::vector<double> got =
        from_device_bf16(dout, static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    const std::string label = "qsa n_kv=" + std::to_string(n_kv) + " T=" + std::to_string(tokens) +
                              " seed=" + std::to_string(c.seed);
    return verify_reduction(label, got, reference, kQsaBf16Criterion);
}
} // namespace

int main() {
    if (cuda_unavailable()) {
        std::printf("no CUDA device; skipping\n");
        return 77;
    }
    std::mt19937 rng(0x51a);
    const Case cases[] = {
        {32, 1, 101}, {32, 3, 102}, {32, 5, 103}, {32, 7, 104},
        {64, 1, 111}, {64, 4, 112}, {64, 8, 113},
        {128, 2, 121}, {128, 6, 122}, {128, 8, 123},
        {9, 3, 131}, {16, 1, 132},
    };
    int failures = 0;
    for (const Case& c : cases) failures += run_case(c, rng);
    // Workspace capacity is zero for the registered geometry; a bad geometry must throw.
    try {
        const std::size_t w = ops::qsa_softmax_attention_workspace_capacity_bytes(
            ninfer::ops::AttentionHeadGeometry{kHeadDim, kQueryHeads, kKVHeads}, 1, 8);
        if (w != 0) {
            std::cerr << "qsa: workspace capacity must be zero\n";
            ++failures;
        }
    } catch (...) {
        std::cerr << "qsa: capacity query unexpectedly threw\n";
        ++failures;
    }
    try {
        [[maybe_unused]] const auto rejected = ops::qsa_softmax_attention_workspace_capacity_bytes(
            ninfer::ops::AttentionHeadGeometry{kHeadDim, kQueryHeads, 4}, 1, 8);
        std::cerr << "qsa: capacity query should reject a non-[256,24,2] geometry\n";
        ++failures;
    } catch (const std::exception&) {
    }

    if (failures) {
        std::cerr << failures << " qsa check(s) failed\n";
        return 1;
    }
    std::printf("qsa_attention: all checks passed\n");
    return 0;
}
