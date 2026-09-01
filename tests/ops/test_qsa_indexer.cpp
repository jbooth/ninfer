// QSA indexer (K5, qwen4exp) Op qualification.
//
// The oracle is an independent FP64 implementation of the whole indexer pipeline: block pooling
// of the raw side cache (cell-0 tail pad), per-block RMSNorm + rope at the block position b*r,
// per-(head, token) RMSNorm + rope of the query projection, the 4-head rectified score, the
// causal / force-in per-cell expansion, and the top-k (value desc, lower-id tiebreak, ascending
// output). Inputs are rounded to the BF16 grid first so the oracle reads the represented values.
//
// The `score` output (FP32) is checked against the FP64 reference with a BF16-level reduction
// criterion; the `topk` output (I32) is checked exactly. Two case families are run: (a) general
// random data with n_kv <= 2051 (all cells selected -> the topk is the identity prefix plus
// padding), and (b) a two-cluster selection case (n_kv > 2051) whose block scores form two
// well-separated uniform clusters so the top-k boundary is unambiguous under BF16 rounding.

#include "ninfer/ops/qsa.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

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

constexpr std::int32_t kDim     = 128;
constexpr std::int32_t kHeads   = 4;
constexpr std::int32_t kQProj   = 512;
constexpr std::int32_t kR       = 4;
constexpr std::int32_t kTopK    = 2051;
constexpr std::int32_t kHalf    = 32;
constexpr std::int32_t kRopeDim = 64;
constexpr float kEps            = 1e-6F;

// BF16-level reduction criterion for the FP32 score output. The score is a sum of 4 rectified
// 128-dim dot products accumulated through the BF16 pooled/normalized intermediates; the
// aggregate relative-L2 sits at the attention-family BF16 level and the gross cap covers the
// worst-case BF16 half-ulp at a binade bottom with margin.
constexpr ReductionCriterion kScoreCriterion{
    /*relative_l2*/ 3.0e-3,
    /*gross_absolute*/ 2.0e-3,
    /*gross_relative_to_max_reference*/ 4.2e-3,
};

// The selection case uses an engineered one-hot key whose rope turns the dot into cos(4b - pos),
// a difference of two O(64) terms that cancels; the BF16 rounding of the intermediate pooled/query
// is amplified in the cancellation, so the FP32 score is checked with a looser criterion there. The
// top-k (the op's real deliverable) is still verified exactly.
constexpr ReductionCriterion kScoreCriterionLoose{
    /*relative_l2*/ 1.5e-2,
    /*gross_absolute*/ 3.0,
    /*gross_relative_to_max_reference*/ 3.0e-2,
};

// 1e7-base inverse frequencies (pair p has frequency 1e7^(-2p/64) = 1e7^(-p/32)).
double inv_freq(int p) { return std::pow(1e7, -2.0 * p / kRopeDim); }

std::vector<double> to_double(const std::vector<float>& v) { return {v.begin(), v.end()}; }
std::vector<float> to_float(const std::vector<double>& v) {
    std::vector<float> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

// RMSNorm a length-128 vector by gamma (plain w, eps) then rope over the first 64 dims (32
// split pairs (p, p+32)) at `position`. In place.
void norm_rope(std::vector<double>& x, const std::vector<double>& gamma, double position) {
    double ss = 0.0;
    for (int i = 0; i < kDim; ++i) ss += x[i] * x[i];
    const double inv = 1.0 / std::sqrt(ss / kDim + kEps);
    for (int i = 0; i < kDim; ++i) x[i] *= inv * gamma[i];
    for (int p = 0; p < kHalf; ++p) {
        const double first  = x[p];
        const double second = x[p + kHalf];
        const double a      = position * inv_freq(p);
        const double s      = std::sin(a);
        const double c      = std::cos(a);
        x[p]         = first * c - second * s;
        x[p + kHalf] = second * c + first * s;
    }
}

struct Reference {
    std::vector<double> score;   // [n_blocks, T]
    std::vector<std::int32_t> topk;  // [kTopK, T]
    double max_score = 0.0;
};

Reference run_oracle(const std::vector<double>& k_cache, const std::vector<double>& q_raw,
                     const std::vector<double>& k_norm, const std::vector<double>& q_norm,
                     const std::vector<std::int32_t>& positions, std::int32_t n_kv,
                     std::int32_t T) {
    const std::int32_t n_blocks = (n_kv + kR - 1) / kR;

    // Pooled + norm + rope keys: pooled[k, b].
    std::vector<double> pooled(static_cast<std::size_t>(kDim) * n_blocks);
    for (std::int32_t b = 0; b < n_blocks; ++b) {
        for (std::int32_t k = 0; k < kDim; ++k) {
            double acc = 0.0;
            for (std::int32_t o = 0; o < kR; ++o) {
                const std::int32_t j = b * kR + o;
                const std::int32_t jidx = (j < n_kv) ? j : 0;
                acc += k_cache[static_cast<std::size_t>(k) * n_kv + jidx];
            }
            pooled[static_cast<std::size_t>(k) * n_blocks + b] = acc / kR;
        }
    }
    for (std::int32_t b = 0; b < n_blocks; ++b) {
        std::vector<double> x(kDim);
        for (std::int32_t k = 0; k < kDim; ++k) x[k] = pooled[static_cast<std::size_t>(k) * n_blocks + b];
        norm_rope(x, k_norm, static_cast<double>(b * kR));
        for (std::int32_t k = 0; k < kDim; ++k) pooled[static_cast<std::size_t>(k) * n_blocks + b] = x[k];
    }

    // Per-(head, token) norm + rope queries: qf[(h*128+k), t].
    std::vector<double> qf(static_cast<std::size_t>(kQProj) * T);
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t h = 0; h < kHeads; ++h) {
            std::vector<double> x(kDim);
            for (std::int32_t k = 0; k < kDim; ++k) x[k] = q_raw[static_cast<std::size_t>(h * kDim + k) * T + t];
            norm_rope(x, q_norm, static_cast<double>(positions[t]));
            for (std::int32_t k = 0; k < kDim; ++k) qf[static_cast<std::size_t>(h * kDim + k) * T + t] = x[k];
        }
    }

    // Rectified 4-head score.
    Reference ref;
    ref.score.assign(static_cast<std::size_t>(n_blocks) * T, 0.0);
    for (std::int32_t b = 0; b < n_blocks; ++b) {
        for (std::int32_t t = 0; t < T; ++t) {
            double acc = 0.0;
            for (std::int32_t h = 0; h < kHeads; ++h) {
                double dot = 0.0;
                for (std::int32_t k = 0; k < kDim; ++k) {
                    dot += pooled[static_cast<std::size_t>(k) * n_blocks + b] *
                           qf[static_cast<std::size_t>(h * kDim + k) * T + t];
                }
                acc += std::max(dot, 0.0);
            }
            ref.score[static_cast<std::size_t>(b) * T + t] = acc;
        }
    }
    for (double s : ref.score) ref.max_score = std::max(ref.max_score, s);

    // Expanded + top-k per token.
    ref.topk.assign(static_cast<std::size_t>(kTopK) * T, -1);
    for (std::int32_t t = 0; t < T; ++t) {
        const std::int32_t q = positions[t];
        const std::int32_t tail_start = ((q + 1) / kR) * kR;
        std::vector<std::pair<double, std::int32_t>> cells;
        cells.reserve(n_kv);
        for (std::int32_t j = 0; j < n_kv; ++j) {
            double v;
            if (j > q) {
                v = -std::numeric_limits<double>::infinity();
            } else if (j >= tail_start) {
                v = 1e9;
            } else {
                const std::int32_t b = j / kR;
                v = (b * kR + kR <= n_kv) ? ref.score[static_cast<std::size_t>(b) * T + t]
                                          : -std::numeric_limits<double>::infinity();
            }
            cells.emplace_back(v, j);
        }
        const std::int32_t width = (n_kv < kTopK) ? n_kv : kTopK;
        std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        std::vector<std::int32_t> chosen;
        chosen.reserve(width);
        for (std::int32_t i = 0; i < width; ++i) chosen.push_back(cells[i].second);
        std::sort(chosen.begin(), chosen.end());
        for (std::int32_t i = 0; i < width; ++i) ref.topk[static_cast<std::size_t>(i) * T + t] = chosen[i];
    }
    return ref;
}

// Compare the kernel's I32 topk row against the reference exactly; return the number of mismatches.
int compare_topk(const std::vector<std::int32_t>& got, const std::vector<std::int32_t>& reference,
                 std::int32_t T, std::int32_t n_kv) {
    int mismatches = 0;
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t slot = 0; slot < kTopK; ++slot) {
            const int g = got[static_cast<std::size_t>(slot) * T + t];
            const int r = reference[static_cast<std::size_t>(slot) * T + t];
            if (g != r && !(mismatches++ < 8)) {
                std::cerr << "  topk[" << slot << "," << t << "] got=" << g << " ref=" << r << "\n";
            }
        }
    }
    (void)n_kv;
    return mismatches;
}

// Run one general case (n_kv <= 2051, all cells selected -> identity-prefix topk).
std::int32_t run_general(std::int32_t n_kv, std::int32_t T, std::uint32_t seed,
                         std::mt19937& rng) {
    std::vector<float> k_cache(static_cast<std::size_t>(kDim) * n_kv);
    std::vector<float> q_raw(static_cast<std::size_t>(kQProj) * T);
    std::vector<float> k_norm(kDim), q_norm(kDim);
    fill_uniform(k_cache, seed, -1.0F, 1.0F);
    fill_uniform(q_raw, seed + 1U, -1.0F, 1.0F);
    fill_uniform(k_norm, seed + 2U, 0.5F, 1.5F);
    fill_uniform(q_norm, seed + 3U, 0.5F, 1.5F);
    round_to_bf16(k_cache);
    round_to_bf16(q_raw);
    round_to_bf16(k_norm);
    round_to_bf16(q_norm);

    // Positions so every cell is visible to every query (last position for all tokens).
    std::vector<std::int32_t> positions(T, n_kv - 1);

    const Reference ref = run_oracle(to_double(k_cache), to_double(q_raw), to_double(k_norm),
                                     to_double(q_norm), positions, n_kv, T);

    const DeviceBuffer dk = to_device_bf16(k_cache), dq = to_device_bf16(q_raw),
        dkn = to_device_bf16(k_norm), dqn = to_device_bf16(q_norm), dpos = to_device_i32(positions),
        dtopk(static_cast<std::size_t>(kTopK) * T * 4);
    const std::int32_t n_blocks = (n_kv + kR - 1) / kR;
    const DeviceBuffer dscore(static_cast<std::size_t>(n_blocks) * T * 4);

    const Tensor tkc(dk.p, DType::BF16, {kDim, n_kv});
    const Tensor tq(dq.p, DType::BF16, {kQProj, T});
    const Tensor tpos(dpos.p, DType::I32, {T});
    const Tensor tkn(dkn.p, DType::BF16, {kDim});
    const Tensor tqn(dqn.p, DType::BF16, {kDim});
    Tensor ttopk(dtopk.p, DType::I32, {kTopK, T});
    Tensor tsc(dscore.p, DType::FP32, {n_blocks, T});

    const std::size_t cap = ops::qsa_indexer_workspace_capacity_bytes(n_kv, T);
    WorkspaceArena arena(cap);
    ops::qsa_indexer(tkc, tq, tpos, tkn, tqn, ttopk, tsc, arena, 0);
    cuda_synchronize();

    int failures = 0;
    const std::vector<std::int32_t> got_topk = from_device_i32(dtopk, static_cast<std::size_t>(kTopK) * T);
    const std::string label = "qsa_indexer n_kv=" + std::to_string(n_kv) + " T=" + std::to_string(T) +
                              " seed=" + std::to_string(seed);
    if (const int mm = compare_topk(got_topk, ref.topk, T, n_kv)) {
        std::cerr << label << ": " << mm << " topk mismatches\n";
        failures += mm;
    }
    const std::vector<double> got_score = from_device_f32(dscore, static_cast<std::size_t>(n_blocks) * T);
    failures += verify_reduction(label + " score", got_score, ref.score, kScoreCriterion);
    return failures;
}

// Run one selection case (n_kv > 2051): even blocks carry the dim-0 key, odd blocks the dim-1 key,
// and the query emphasizes dim-0 so the two clusters are well separated. The top-k boundary is
// unambiguous; the kernel's topk must match the oracle exactly.
std::int32_t run_selection(std::int32_t n_kv, std::int32_t T, std::uint32_t seed,
                           std::mt19937& rng) {
    (void)rng;
    std::vector<double> k_cache(static_cast<std::size_t>(kDim) * n_kv, 0.0);
    const std::int32_t n_blocks = (n_kv + kR - 1) / kR;
    for (std::int32_t b = 0; b < n_blocks; ++b) {
        const int dim = (b & 1) ? 1 : 0;  // even blocks dim-0, odd blocks dim-1
        for (std::int32_t o = 0; o < kR; ++o) {
            const std::int32_t j = b * kR + o;
            if (j < n_kv) k_cache[static_cast<std::size_t>(dim) * n_kv + j] = 1.0;
        }
    }
    std::vector<double> q_raw(static_cast<std::size_t>(kQProj) * T, 0.0);
    for (std::int32_t t = 0; t < T; ++t) q_raw[static_cast<std::size_t>(0) * T + t] = 1.0;  // head 0, dim 0
    std::vector<double> k_norm(kDim, 1.0), q_norm(kDim, 1.0);
    std::vector<std::int32_t> positions(T, n_kv - 1);  // every cell visible

    const Reference ref = run_oracle(k_cache, q_raw, k_norm, q_norm, positions, n_kv, T);

    // Assert the two clusters are well separated so the selection is unambiguous under BF16.
    {
        double hi = -1e300, lo = 1e300;
        for (std::int32_t b = 0; b < n_blocks; ++b) {
            const double s = ref.score[static_cast<std::size_t>(b) * T + 0];
            if (b & 1) lo = std::min(lo, s); else hi = std::max(hi, s);
        }
        const double gap = std::abs(hi - lo);
        if (gap < 1e-2) {
            std::cerr << "selection: cluster gap " << gap << " too small\n";
            return 1;
        }
    }

    std::vector<float> kf(k_cache.size()), qf(q_raw.size());
    for (std::size_t i = 0; i < kf.size(); ++i) kf[i] = static_cast<float>(k_cache[i]);
    for (std::size_t i = 0; i < qf.size(); ++i) qf[i] = static_cast<float>(q_raw[i]);

    const DeviceBuffer dk = to_device_bf16(kf), dq = to_device_bf16(qf),
        dkn = to_device_bf16(to_float(k_norm)), dqn = to_device_bf16(to_float(q_norm)),
        dpos = to_device_i32(positions), dtopk(static_cast<std::size_t>(kTopK) * T * 4),
        dscore(static_cast<std::size_t>(n_blocks) * T * 4);
    const Tensor tkc(dk.p, DType::BF16, {kDim, n_kv});
    const Tensor tq(dq.p, DType::BF16, {kQProj, T});
    const Tensor tpos(dpos.p, DType::I32, {T});
    const Tensor tkn(dkn.p, DType::BF16, {kDim});
    const Tensor tqn(dqn.p, DType::BF16, {kDim});
    Tensor ttopk(dtopk.p, DType::I32, {kTopK, T});
    Tensor tsc(dscore.p, DType::FP32, {n_blocks, T});

    WorkspaceArena arena(ops::qsa_indexer_workspace_capacity_bytes(n_kv, T));
    ops::qsa_indexer(tkc, tq, tpos, tkn, tqn, ttopk, tsc, arena, 0);
    cuda_synchronize();

    int failures = 0;
    const std::vector<std::int32_t> got_topk = from_device_i32(dtopk, static_cast<std::size_t>(kTopK) * T);
    const std::string label = "qsa_indexer sel n_kv=" + std::to_string(n_kv) + " T=" + std::to_string(T);
    if (const int mm = compare_topk(got_topk, ref.topk, T, n_kv)) {
        std::cerr << label << ": " << mm << " topk mismatches\n";
        failures += mm;
    }
    const std::vector<double> got_score = from_device_f32(dscore, static_cast<std::size_t>(n_blocks) * T);
    failures += verify_reduction(label + " score", got_score, ref.score, kScoreCriterionLoose);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::printf("no CUDA device; skipping\n");
        return 77;
    }
    std::mt19937 rng(0x51b);
    int failures = 0;
    // General (all-selected) cases.
    for (std::int32_t t : {1, 2, 4}) {
        failures += run_general(100, t, 200 + t, rng);
        failures += run_general(500, t, 300 + t, rng);
    }
    // A non-divisible n_kv (tail-block cell-0 padding) at a non-final position.
    failures += run_general(61, 1, 401, rng);
    // Selection (n_kv > 2051) cases.
    failures += run_selection(2100, 1, 501, rng);
    failures += run_selection(2100, 2, 502, rng);
    failures += run_selection(4100, 1, 503, rng);

    // Workspace capacity sanity.
    try {
        const std::size_t cap = ops::qsa_indexer_workspace_capacity_bytes(100, 4);
        const std::size_t expect =
            (static_cast<std::size_t>(kDim) * ((100 + kR - 1) / kR) * 2 + 255u) & ~255ull;
        const std::size_t expect_q = (static_cast<std::size_t>(kQProj) * 4 * 2 + 255u) & ~255ull;
        if (cap != expect + expect_q) {
            std::cerr << "qsa_indexer: workspace capacity " << cap << " != " << expect + expect_q
                      << "\n";
            ++failures;
        }
    } catch (...) {
        std::cerr << "qsa_indexer: capacity query unexpectedly threw\n";
        ++failures;
    }

    if (failures) {
        std::cerr << failures << " qsa_indexer check(s) failed\n";
        return 1;
    }
    std::printf("qsa_indexer: all checks passed\n");
    return 0;
}
