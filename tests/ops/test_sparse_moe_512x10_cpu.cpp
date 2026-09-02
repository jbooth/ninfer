// tests/ops/test_sparse_moe_512x10_cpu.cpp — closed-Op test for the host CPU route of the
// Qwen4Exp 512-expert top-10 routed MoE (jbq4). The oracle is the same FP64 MoE formula the
// GPU route (K7) is verified against (llama.cpp qwen4exp.cpp::build_layer_ffn), evaluated on
// the exact logical dequant weights. The CPU op's implementation-profile differences from the
// oracle are the int8 activation quantization (input once per token, and each expert's swiglu
// intermediate) and the FP32 accumulation; the criterion absorbs both. The synthetic bank and
// the oracle are byte-identical to test_sparse_moe_512x10 so both routes share the same weights.
//
// Gated here: (1) CPU vs FP64 oracle for both codecs (Q4 + Q8), T in {1,2}; (2) determinism —
// 1-thread vs N-thread runs are bit-identical.
#include "ninfer/ops/sparse_moe_512x10_cpu.h"
#include "op_check.h"
#include "op_tester.h"
#include "quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

struct RoutePattern {
    std::int32_t selected[kTopK];
};
constexpr RoutePattern kPatterns[2] = {
    {{15, 14, 13, 12, 11, 10, 9, 8, 7, 6}},
    {{9, 11, 13, 15, 0, 1, 2, 3, 4, 5}},
};

inline double bf16d(std::uint16_t v) { return static_cast<double>(bf16_to_f32(v)); }
inline double silu64(double x) { return x / (1.0 + std::exp(-x)); }
inline double sigmoid64(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double bf16i(std::int16_t v) {  // bf16 (as int16 bits) -> double
    return static_cast<double>(bf16_to_f32(static_cast<std::uint16_t>(v)));
}

static double dot_fp64(const float* w, std::int64_t row, const std::uint16_t* x, std::int32_t k,
                       std::int32_t T, int t) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) { s += static_cast<double>(w[row * k + kk]) * bf16d(x[kk * T + t]); }
    return s;
}
static double dot_act_fp64(const float* w, std::int64_t row, const double* a, std::int32_t k) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) { s += static_cast<double>(w[row * k + kk]) * a[kk]; }
    return s;
}
static double dot_router_fp32(const float* w, std::int64_t row, const std::uint16_t* x,
                              std::int32_t k, std::int32_t T, int t) {
    double s = 0.0;
    for (int kk = 0; kk < k; ++kk) { s += static_cast<double>(w[row * k + kk]) * bf16d(x[kk * T + t]); }
    return s;
}

struct Sources {
    std::vector<float> gate_up;
    std::vector<float> down;
    std::vector<float> shared_gu;
    std::vector<float> shared_down;
    std::vector<std::uint16_t> input;
    std::vector<float> router;
    std::vector<std::uint16_t> residual;
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
    s.input.resize(static_cast<std::size_t>(kHidden) * kMaxT);
    // Realistic dynamic range (all elements ~0.05-0.1) so the W4A8 per-token int8 activation
    // quantization operates in its normal regime (amax ~= a few x typical element).
    for (int t = 0; t < kMaxT; ++t) {
        const int pattern = t % 2;
        for (int col = 0; col < kHidden; ++col) {
            float val;
            if (col == 0) {
                val = 0.1f;  // router bias carrier
            } else if (col == kHidden - 2) {
                val = (pattern == 0) ? 0.1f : 0.0f;
            } else if (col == kHidden - 1) {
                val = (pattern == 1) ? 0.1f : 0.0f;
            } else {
                val = 0.07f * std::sin(0.11f * col + 0.05f * t + 0.3f);
            }
            s.input[static_cast<std::size_t>(col) * kMaxT + t] = f32_to_bf16(val);
        }
    }
    s.router.resize(static_cast<std::size_t>(kExperts + 1) * kHidden);
    for (int row = 0; row <= kExperts; ++row) {
        for (int col = 0; col < kHidden; ++col) {
            s.router[static_cast<std::size_t>(row) * kHidden + col] =
                (static_cast<float>((col * 13 + 5) % 17) - 8.0f) * 0.0001f;
        }
    }
    for (int e = 0; e < kExperts; ++e) { s.router[static_cast<std::size_t>(e) * kHidden + 0] -= 80.0f; }
    s.router[static_cast<std::size_t>(kExperts) * kHidden + 0] += 3.75f;  // shared gate
    for (int p = 0; p < 2; ++p) {
        const int marker = kHidden - 2 + p;
        for (int rank = 0; rank < kTopK; ++rank) {
            const float base = 4.0f - 0.5f * rank;
            const int e      = kPatterns[p].selected[rank];
            s.router[static_cast<std::size_t>(e) * kHidden + marker] += (base + 8.0f) * 10.0f;
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
                std::cerr << "oracle: selection mismatch at token " << t << " rank " << rank << "\n";
                throw std::runtime_error("top-k selection mismatch");
            }
        }
        for (int rank = 0; rank < kTopK; ++rank) {
            topk_w[rank] = std::exp(scores[order[rank]] - mx) / sum;
        }
        double wsum = 0.0;
        for (int rank = 0; rank < kTopK; ++rank) { wsum += topk_w[rank]; }
        const double denom = std::max(wsum, 6.103515625e-5);
        for (int rank = 0; rank < kTopK; ++rank) { topk_w[rank] /= denom; }
        for (int rank = 0; rank < kTopK; ++rank) {
            const int e = order[rank];
            for (int j = 0; j < kInter; ++j) {
                const double gate =
                    dot_fp64(gate_up.dequant.data(), static_cast<std::int64_t>(e) * 2 * kInter + j, x, kHidden, T, t);
                const double up =
                    dot_fp64(gate_up.dequant.data(), static_cast<std::int64_t>(e) * 2 * kInter + kInter + j, x, kHidden, T, t);
                act[j] = silu64(gate) * up;
            }
            for (int i = 0; i < kHidden; ++i) {
                y[rank * kHidden + i] =
                    dot_act_fp64(down.dequant.data(), static_cast<std::int64_t>(e) * kHidden + i, act.data(), kInter);
            }
        }
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

template <int Ts>
void check_cpu(const Sources& s, QType qtype, ops::MoeCode code, const char* label) {
    auto gate_up = quantized_weight::pack_row_split_lowbit(s.gate_up, kExperts * 2 * kInter, kHidden, qtype);
    auto down    = quantized_weight::pack_row_split_lowbit(s.down, kExperts * kHidden, kInter, qtype);
    auto sgu     = quantized_weight::pack_row_split_lowbit(s.shared_gu, 2 * kInter, kHidden, qtype);
    auto sdown   = quantized_weight::pack_row_split_lowbit(s.shared_down, kHidden, kInter, qtype);

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
    std::vector<std::int16_t> x_host(x_dense.size());
    std::memcpy(x_host.data(), x_dense.data(), x_dense.size() * 2);
    auto seed = [&]() {
        std::vector<std::int16_t> d(res_dense.size());
        std::memcpy(d.data(), res_dense.data(), res_dense.size() * 2);
        return d;
    };

    ops::SparseMoe512x10Geometry geo;
    geo.n_experts = kExperts;
    geo.top_k     = kTopK;
    geo.hidden    = kHidden;
    geo.inter     = kInter;

    ops::SparseMoe512x10CpuWeights w;
    w.router = s.router.data();
    w.routed_gate_up_codec = code;
    w.routed_gate_up_base = gate_up.payload.data();
    w.routed_gate_up_scales = reinterpret_cast<const std::uint16_t*>(gate_up.payload.data() + gate_up.scale_plane_offset);
    w.routed_down_codec = code;
    w.routed_down_base = down.payload.data();
    w.routed_down_scales = reinterpret_cast<const std::uint16_t*>(down.payload.data() + down.scale_plane_offset);
    w.shared_gate_up_codec = code;
    w.shared_gate_up_base = sgu.payload.data();
    w.shared_gate_up_scales = reinterpret_cast<const std::uint16_t*>(sgu.payload.data() + sgu.scale_plane_offset);
    w.shared_down_codec = code;
    w.shared_down_base = sdown.payload.data();
    w.shared_down_scales = reinterpret_cast<const std::uint16_t*>(sdown.payload.data() + sdown.scale_plane_offset);

    // Determinism: 1-thread vs N-thread runs are bit-identical.
    auto dest1 = seed();
    ops::sparse_moe_512x10_cpu(x_host.data(), Ts, geo, w, dest1.data(), 1);
    auto dest2 = seed();
    ops::sparse_moe_512x10_cpu(x_host.data(), Ts, geo, w, dest2.data(), 8);
    if (!std::equal(dest1.begin(), dest1.end(), dest2.begin())) {
        std::cerr << "FAILED determinism " << label << " T=" << Ts << "\n";
        std::exit(1);
    }

    std::vector<double> ref;
    oracle(s, Ts, x_dense, res_dense, gate_up, down, sgu, sdown, ref);
    std::vector<double> got(dest1.size());
    for (std::size_t i = 0; i < dest1.size(); ++i) { got[i] = bf16i(dest1[i]); }

    // W4A8 activation error on a realistic-dynamic-range input (amax ~= few x typical) is ~1e-4
    // relL2 (the exact int32 per-group dot avoids any BF16 staging error); the criterion has
    // ~10x headroom for input variation.
    constexpr ReductionCriterion criterion{1.0e-3, 1e-30, 2.0e-4};
    const auto stats = compute_reduction_stats(got.data(), ref.data(),
                                               static_cast<std::int64_t>(kHidden) * Ts);
    std::cerr << label << " T=" << Ts << " relL2=" << stats.relative_l2
              << " maxRef=" << stats.maximum_absolute_reference
              << " maxErr=" << stats.maximum_absolute_error << " (deterministic)\n";
    if (!reduction_passes(stats, static_cast<std::int64_t>(kHidden) * Ts, criterion)) {
        std::cerr << "FAILED " << label << " T=" << Ts << " maxErr idx " << stats.maximum_error_index
                  << " got=" << stats.actual_at_maximum << " ref=" << stats.reference_at_maximum << "\n";
        std::exit(1);
    }
    std::cerr << "PASS " << label << " T=" << Ts << "\n";
}

}  // namespace

int main() {
    const char* skip = std::getenv("NINFER_SKIP_TESTS");
    if (skip && *skip) {
        std::cerr << "ninfer_sparse_moe_512x10_cpu_test: skipped\n";
        return 77;
    }
    const Sources src = make_sources();
    check_cpu<1>(src, QType::Q4G64_F16S, ops::MoeCode::Q4G64, "Q4");
    check_cpu<2>(src, QType::Q4G64_F16S, ops::MoeCode::Q4G64, "Q4");
    check_cpu<1>(src, QType::Q8G64_F16S, ops::MoeCode::Q8G64, "Q8");
    check_cpu<2>(src, QType::Q8G64_F16S, ops::MoeCode::Q8G64, "Q8");
    std::cerr << "all sparse_moe_512x10_cpu checks passed\n";
    return 0;
}
