// tests/ops/test_sparse_moe_512x10.cpp — closed-Op test for the Qwen4Exp 512-expert
// top-10 routed MoE (jbq4), GPU prefill route.
//
// The oracle is the complete FP64 MoE formula (llama.cpp qwen4exp.cpp::build_layer_ffn)
// evaluated on the EXACT logical dequant weights (code * exact FP16 scale). The routed and
// shared MoE banks use the row-split-k128-v1, group-64, FP16-scale layout with two codecs:
// Q4G64 (packed nibbles, 32 B/group) and Q8G64 (raw i8, 64 B/group). The router is FP32.
// The kernel's only implementation-profile differences from the oracle are FP32 accumulation
// and the BF16 staging of the intermediate activations; the reduction criterion absorbs both.
//
// A reduced 16-expert bank keeps the fixtures small enough to run on a partially occupied
// GPU; the 512-expert production geometry is exercised end-to-end at JM6.
#include "core/arena.h"
#include "ninfer/ops/sparse_moe_512x10.h"
#include "op_check.h"
#include "op_tester.h"
#include "quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kInter   = 640;
constexpr std::int32_t kExperts = 16;
constexpr std::int32_t kTopK    = 10;
constexpr std::int32_t kMaxT    = 4;

// Expert ids in score-descending order (topk order) for each of the two route patterns.
// Across the two patterns every one of the 16 experts is selected at least once.
struct RoutePattern {
    std::int32_t selected[kTopK];
};
constexpr RoutePattern kPatterns[2] = {
    {{15, 14, 13, 12, 11, 10, 9, 8, 7, 6}},
    {{9, 11, 13, 15, 0, 1, 2, 3, 4, 5}},
};

struct QuantGeometry {
    std::uint8_t group;
    std::uint8_t code_bytes;  // per group
};

inline QuantGeometry quant_geometry(QType q) {
    switch (q) {
    case QType::Q4G64_F16S: return {64, 32};
    case QType::Q8G64_F16S: return {64, 64};
    default: throw std::invalid_argument("unsupported qtype");
    }
}

inline double bf16d(std::uint16_t v) { return static_cast<double>(bf16_to_f32(v)); }
inline double silu64(double x) { return x / (1.0 + std::exp(-x)); }
inline double sigmoid64(double x) { return 1.0 / (1.0 + std::exp(-x)); }

static double dot_fp64(const float* w, std::int64_t row, const std::uint16_t* x, std::int32_t k,
                       std::int32_t T, int t) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) {
        s += static_cast<double>(w[row * k + kk]) * bf16d(x[kk * T + t]);
    }
    return s;
}

static double dot_act_fp64(const float* w, std::int64_t row, const double* a, std::int32_t k) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) { s += static_cast<double>(w[row * k + kk]) * a[kk]; }
    return s;
}

// FP32 router row @ BF16 activation column, accumulated in FP64 (exact router values).
static double dot_router_fp32(const float* w, std::int64_t row, const std::uint16_t* x,
                              std::int32_t k, std::int32_t T, int t) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) { s += static_cast<double>(w[row * k + kk]) * bf16d(x[kk * T + t]); }
    return s;
}

// ---- deterministic inputs (codec-independent) -----------------------------------------
struct Sources {
    std::vector<float> gate_up;      // [kExperts*2*inter, kHidden]
    std::vector<float> down;         // [kExperts*kHidden, kInter]
    std::vector<float> shared_gu;    // [2*inter, kHidden]
    std::vector<float> shared_down;  // [kHidden, kInter]
    std::vector<std::uint16_t> input;    // [kHidden, kMaxT] bf16
    std::vector<float> router;           // [kExperts+1, kHidden]
    std::vector<std::uint16_t> residual; // [kHidden, kMaxT] bf16
};

Sources make_sources() {
    Sources s;
    s.gate_up.resize(static_cast<std::size_t>(kExperts) * 2 * kInter * kHidden);
    s.down.resize(static_cast<std::size_t>(kExperts) * kHidden * kInter);
    s.shared_gu.resize(static_cast<std::size_t>(2 * kInter) * kHidden);
    s.shared_down.resize(static_cast<std::size_t>(kHidden) * kInter);
    for (int e = 0; e < kExperts; ++e) {
        for (int j = 0; j < kInter; ++j) {
            for (int k = 0; k < kHidden; ++k) {
                s.gate_up[(e * 2 * kInter + j) * kHidden + k] =
                    0.011f * std::sin(0.01f * e + 0.031f * j + 0.0021f * k);
                s.gate_up[(e * 2 * kInter + kInter + j) * kHidden + k] =
                    0.011f * std::sin(0.013f * e + 0.029f * j + 0.0019f * k + 2.0f);
            }
        }
        for (int i = 0; i < kHidden; ++i) {
            for (int k = 0; k < kInter; ++k) {
                s.down[(e * kHidden + i) * kInter + k] =
                    0.0056f * std::sin(0.02f * e + 0.011f * i + 0.0017f * k + 0.7f);
            }
        }
    }
    for (int j = 0; j < kInter; ++j) {
        for (int k = 0; k < kHidden; ++k) {
            s.shared_gu[j * kHidden + k] = 0.011f * std::sin(50.0f + 0.031f * j + 0.0021f * k);
            s.shared_gu[(kInter + j) * kHidden + k] =
                0.011f * std::sin(65.0f + 0.029f * j + 0.0019f * k + 2.0f);
        }
    }
    for (int i = 0; i < kHidden; ++i) {
        for (int k = 0; k < kInter; ++k) {
            s.shared_down[i * kInter + k] = 0.0056f * std::sin(100.0f + 0.011f * i + 0.0017f * k + 0.7f);
        }
    }

    // Input: per token, x[0]=1.0 (bias carrier), two marker columns, rest small.
    s.input.resize(static_cast<std::size_t>(kHidden) * kMaxT);
    for (int t = 0; t < kMaxT; ++t) {
        const int pattern = t % 2;
        for (int col = 0; col < kHidden; ++col) {
            float val;
            if (col == 0) {
                val = 1.0f;
            } else if (col == kHidden - 2) {
                val = (pattern == 0) ? 1.0f : 0.0f;
            } else if (col == kHidden - 1) {
                val = (pattern == 1) ? 1.0f : 0.0f;
            } else {
                val = 0.025f + static_cast<float>((col * 7 + pattern * 11) % 19) * 0.002f;
            }
            s.input[static_cast<std::size_t>(col) * kMaxT + t] = f32_to_bf16(val);
        }
    }

    // Router: tiny generic entries, -8.0 base bias via x[0], per-pattern marker boosts.
    s.router.resize(static_cast<std::size_t>(kExperts + 1) * kHidden);
    for (int row = 0; row <= kExperts; ++row) {
        for (int col = 0; col < kHidden; ++col) {
            s.router[static_cast<std::size_t>(row) * kHidden + col] =
                (static_cast<float>((col * 13 + 5) % 17) - 8.0f) * 0.0001f;
        }
    }
    for (int e = 0; e < kExperts; ++e) {
        s.router[static_cast<std::size_t>(e) * kHidden + 0] -= 8.0f;
    }
    s.router[static_cast<std::size_t>(kExperts) * kHidden + 0] += 0.375f;  // shared gate
    for (int p = 0; p < 2; ++p) {
        const int marker = kHidden - 2 + p;
        for (int rank = 0; rank < kTopK; ++rank) {
            const float base = 4.0f - 0.5f * rank;
            const int e      = kPatterns[p].selected[rank];
            s.router[static_cast<std::size_t>(e) * kHidden + marker] += base + 8.0f;
        }
    }

    s.residual.resize(static_cast<std::size_t>(kHidden) * kMaxT);
    for (int i = 0; i < kHidden; ++i) {
        for (int t = 0; t < kMaxT; ++t) {
            s.residual[static_cast<std::size_t>(i) * kMaxT + t] =
                f32_to_bf16(0.2f * std::sin(0.11f * i + 0.05f * t + 0.3f));
        }
    }
    return s;
}

// FP64 reference for the complete MoE formula on the exact logical dequant weights.
template <typename Pack>
void oracle(const Sources& s, int T, const std::vector<std::uint16_t>& x_dense,
            const std::vector<std::uint16_t>& res_dense, const Pack& gate_up, const Pack& down,
            const Pack& sgu, const Pack& sdown, std::vector<double>& ref) {
    ref.assign(static_cast<std::size_t>(kHidden) * T, 0.0);
    std::vector<double> scores(kExperts + 1);
    std::vector<double> prob(kExperts);
    std::vector<double> y(kTopK * kHidden);
    std::vector<double> act(kInter);
    std::vector<double> sact(kInter);
    std::vector<double> topk_w(kTopK);
    const std::uint16_t* x = x_dense.data();
    for (int t = 0; t < T; ++t) {
        const int pattern  = t % 2;
        const auto& intent = kPatterns[pattern].selected;
        for (int e = 0; e <= kExperts; ++e) {
            scores[e] = dot_router_fp32(s.router.data(), e, x, kHidden, T, t);
        }
        double mx = -1e300;
        for (int e = 0; e < kExperts; ++e) { mx = std::max(mx, scores[e]); }
        double sum = 0.0;
        for (int e = 0; e < kExperts; ++e) { sum += std::exp(scores[e] - mx); }
        for (int e = 0; e < kExperts; ++e) { prob[e] = std::exp(scores[e] - mx) / sum; }
        // top-10 (prob desc, lower-id asc)
        std::vector<int> order;
        for (int rank = 0; rank < kTopK; ++rank) {
            int best  = -1;
            double bestp = -1.0;
            for (int e = 0; e < kExperts; ++e) {
                if (best < 0 || prob[e] > bestp || (prob[e] == bestp && e < best)) {
                    best = e;
                    bestp = prob[e];
                }
            }
            order.push_back(best);
            prob[best] = -1.0;
        }
        for (int rank = 0; rank < kTopK; ++rank) {
            if (order[rank] != intent[rank]) {
                std::cerr << "oracle: selection mismatch at token " << t << " rank " << rank
                          << " got " << order[rank] << " want " << intent[rank] << "\n";
                throw std::runtime_error("top-k selection mismatch");
            }
        }
        for (int rank = 0; rank < kTopK; ++rank) { topk_w[rank] = std::exp(scores[order[rank]] - mx) / sum; }
        double wsum = 0.0;
        for (int rank = 0; rank < kTopK; ++rank) { wsum += topk_w[rank]; }
        const double denom = std::max(wsum, 6.103515625e-5);
        for (int rank = 0; rank < kTopK; ++rank) { topk_w[rank] /= denom; }

        // routed experts
        for (int rank = 0; rank < kTopK; ++rank) {
            const int e = order[rank];
            for (int j = 0; j < kInter; ++j) {
                const double gate = dot_fp64(gate_up.dequant.data(), static_cast<std::int64_t>(e) * 2 * kInter + j, x, kHidden, T, t);
                const double up = dot_fp64(gate_up.dequant.data(), static_cast<std::int64_t>(e) * 2 * kInter + kInter + j, x, kHidden, T, t);
                act[j] = silu64(gate) * up;
            }
            for (int i = 0; i < kHidden; ++i) {
                y[rank * kHidden + i] =
                    dot_act_fp64(down.dequant.data(), static_cast<std::int64_t>(e) * kHidden + i, act.data(), kInter);
            }
        }
        // shared expert
        for (int j = 0; j < kInter; ++j) {
            const double gate = dot_fp64(sgu.dequant.data(), j, x, kHidden, T, t);
            const double up = dot_fp64(sgu.dequant.data(), kInter + j, x, kHidden, T, t);
            sact[j] = silu64(gate) * up;
        }
        for (int i = 0; i < kHidden; ++i) {
            const double sy = dot_act_fp64(sdown.dequant.data(), i, sact.data(), kInter);
            double acc = 0.0;
            for (int rank = 0; rank < kTopK; ++rank) { acc += topk_w[rank] * y[rank * kHidden + i]; }
            const double gated = sigmoid64(scores[kExperts]) * sy;
            const double res   = bf16d(res_dense[static_cast<std::size_t>(i) * T + t]);
            ref[static_cast<std::size_t>(i) * T + t] = res + acc + gated;
        }
    }
}

// Copy one PackedWeight's planes (codes + scales, whole bank) to device.
struct DeviceBank {
    DeviceBuffer codes;
    DeviceBuffer scales;
    explicit DeviceBank(const quantized_weight::PackedWeight& p)
        : codes(p.code_plane_bytes), scales(p.scale_plane_bytes) {
        codes.copy_from_host(p.payload.data(), p.code_plane_bytes, 0);
        scales.copy_from_host(p.payload.data() + p.scale_plane_offset, p.scale_plane_bytes, 0);
    }
};

template <typename Pack, int Ts>
void check(const Sources& s, QType qtype, ops::MoeCode code, const char* label) {
    Pack gate_up = quantized_weight::pack_row_split_lowbit(s.gate_up, kExperts * 2 * kInter, kHidden, qtype);
    Pack down    = quantized_weight::pack_row_split_lowbit(s.down, kExperts * kHidden, kInter, qtype);
    Pack sgu     = quantized_weight::pack_row_split_lowbit(s.shared_gu, 2 * kInter, kHidden, qtype);
    Pack sdown   = quantized_weight::pack_row_split_lowbit(s.shared_down, kHidden, kInter, qtype);

    DeviceBank gu(gate_up), dn(down), sg(sgu), sd(sdown);

    // Build dense [kHidden, Ts] views of the first Ts tokens from the kMaxT source.
    std::vector<std::uint16_t> x_dense(static_cast<std::size_t>(kHidden) * Ts);
    std::vector<std::uint16_t> res_dense(static_cast<std::size_t>(kHidden) * Ts);
    for (int col = 0; col < kHidden; ++col) {
        for (int t = 0; t < Ts; ++t) {
            x_dense[static_cast<std::size_t>(col) * Ts + t] =
                s.input[static_cast<std::size_t>(col) * kMaxT + t];
            res_dense[static_cast<std::size_t>(col) * Ts + t] =
                s.residual[static_cast<std::size_t>(col) * kMaxT + t];
        }
    }
    DeviceBuffer x_d(static_cast<std::size_t>(kHidden) * Ts * 2);
    x_d.copy_from_host(x_dense.data(), static_cast<std::size_t>(kHidden) * Ts * 2, 0);
    DeviceBuffer dest_d(static_cast<std::size_t>(kHidden) * Ts * 2);
    dest_d.copy_from_host(res_dense.data(), static_cast<std::size_t>(kHidden) * Ts * 2, 0);
    DeviceBuffer router_d(static_cast<std::size_t>(kExperts + 1) * kHidden * 4);
    router_d.copy_from_host(s.router.data(), router_d.bytes, 0);

    DeviceArena arena(1 << 24);
    Tensor x{static_cast<void*>(x_d.p), DType::BF16, {kHidden, Ts}};
    Tensor dest{static_cast<void*>(dest_d.p), DType::BF16, {kHidden, Ts}};

    ops::SparseMoe512x10Geometry geo;
    geo.n_experts = kExperts;
    geo.top_k     = kTopK;
    geo.hidden    = kHidden;
    geo.inter     = kInter;

    ops::SparseMoe512x10Weights w;
    w.router               = static_cast<const float*>(router_d.p);
    w.routed_gate_up_codes = gu.codes.p;
    w.routed_gate_up_scales = gu.scales.p;
    w.routed_down_codes    = dn.codes.p;
    w.routed_down_scales   = dn.scales.p;
    w.shared_gate_up_codes = sg.codes.p;
    w.shared_gate_up_scales = sg.scales.p;
    w.shared_down_codes    = sd.codes.p;
    w.shared_down_scales   = sd.scales.p;
    w.code = code;

    ops::sparse_moe_512x10(x, geo, w, ops::SparseMoe512x10Epilogue::AddResidual, dest, arena,
                           /*stream=*/nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "sparse_moe_512x10 sync");

    std::vector<double> ref;
    oracle(s, Ts, x_dense, res_dense, gate_up, down, sgu, sdown, ref);

    std::vector<std::uint16_t> raw(static_cast<std::size_t>(kHidden) * Ts);
    dest_d.copy_to_host(raw.data(), raw.size() * 2, 0);
    std::vector<double> got(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) { got[i] = bf16d(raw[i]); }

    constexpr test::ReductionCriterion criterion{1.2e-2, 1e-30, 4e-3};
    const auto stats = test::compute_reduction_stats(
        got.data(), ref.data(), static_cast<std::int64_t>(kHidden) * Ts);
    std::cerr << label << " T=" << Ts << " relL2=" << stats.relative_l2
              << " rmse=" << stats.root_mean_squared_error
              << " maxRef=" << stats.maximum_absolute_reference
              << " maxErr=" << stats.maximum_absolute_error
              << " peakWorkspace=" << arena.peak_used() << "B\n";
    if (!test::reduction_passes(stats, static_cast<std::int64_t>(kHidden) * Ts, criterion)) {
        std::cerr << "FAILED " << label << " T=" << Ts << " maxErr at idx "
                  << stats.maximum_error_index << " got=" << stats.actual_at_maximum
                  << " ref=" << stats.reference_at_maximum << "\n";
        std::exit(1);
    }
    std::cerr << "PASS " << label << " T=" << Ts << "\n";
}

}  // namespace

int main() {
    const char* skip = std::getenv("NINFER_SKIP_TESTS");
    if (skip && *skip) {
        std::cerr << "ninfer_sparse_moe_512x10_test: skipped\n";
        return 77;
    }

    const Sources src = make_sources();

    check<quantized_weight::PackedWeight, 1>(src, QType::Q4G64_F16S, ops::MoeCode::Q4G64, "Q4");
    check<quantized_weight::PackedWeight, 2>(src, QType::Q4G64_F16S, ops::MoeCode::Q4G64, "Q4");
    check<quantized_weight::PackedWeight, 4>(src, QType::Q4G64_F16S, ops::MoeCode::Q4G64, "Q4");
    check<quantized_weight::PackedWeight, 1>(src, QType::Q8G64_F16S, ops::MoeCode::Q8G64, "Q8");
    check<quantized_weight::PackedWeight, 2>(src, QType::Q8G64_F16S, ops::MoeCode::Q8G64, "Q8");
    check<quantized_weight::PackedWeight, 4>(src, QType::Q8G64_F16S, ops::MoeCode::Q8G64, "Q8");

    std::cerr << "all sparse_moe_512x10 checks passed\n";
    return 0;
}
