// M2 gate (phase3.md §4.4): numerical qualification of the qwen4exp hyper-connection kernels
// (include/ninfer/ops/hc_mix.h) against naive FP64 oracles of the represented inputs.
//
// Geometry: the real qwen4exp HC shape (D=10240, stream=2560, hc=4, lr=320) plus small
// synthetic shapes, over T in {1, 2, 3, 5, 17} (decode and ragged prefill widths).

#include "ninfer/ops/hc_mix.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-6F;

// Criterion for these BF16-output kernels: the oracle evaluates the complete FP64 formula and
// the only implementation error is the BF16 storage rounding of the output (kernel-internal
// FP32 error is ~1e-6, negligible). Worst-case half-ulp at the bottom of a binade is 2^-8 =
// 3.906e-3 relative, so the gross term is 4.2e-3 (10% margin) and the relative-L2 term absorbs
// the random distribution of per-element storage rounding.
ReductionCriterion hc_bf16_criterion() {
    return {/*relative_l2*/ 2.4e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 4.2e-3};
}

std::vector<double> to_doubles(const DeviceBuffer& buf, std::size_t n) {
    const std::vector<std::uint16_t> h = from_device<std::uint16_t>(buf, n);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<double>(bf16_to_f32(h[i]));
    }
    return out;
}

DeviceBuffer to_device_f32(const std::vector<float>& h) {
    std::vector<std::uint32_t> bits(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        std::memcpy(&bits[i], &h[i], 4);
    }
    return to_device<std::uint32_t>(bits);
}

// x: [D, T] row-major (element [i,t] at i*T+t), already on the bf16 grid.
DeviceBuffer to_device_bf16_2d(const std::vector<float>& h) {
    std::vector<std::uint16_t> bits(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        bits[i] = f32_to_bf16(h[i]);
    }
    return to_device<std::uint16_t>(bits);
}

int run_grouped_rmsnorm(std::int32_t d, std::int32_t stream_dim, std::int32_t t, std::uint32_t seed) {
    const std::int32_t hc = d / stream_dim;
    std::vector<float> x(static_cast<std::size_t>(d) * t), gamma(d);
    fill_uniform(x, seed, -4.0F, 4.0F);
    round_to_bf16(x);
    fill_uniform(gamma, seed + 1U, 0.85F, 1.9F); // folded 1 + w

    std::vector<double> ref(static_cast<std::size_t>(d) * t);
    for (std::int32_t s = 0; s < hc; ++s) {
        for (std::int32_t tok = 0; tok < t; ++tok) {
            double sum = 0.0;
            for (std::int32_t i2 = 0; i2 < stream_dim; ++i2) {
                const double v = x[static_cast<std::size_t>(s * stream_dim + i2) * t + tok];
                sum += v * v;
            }
            const double inv = 1.0 / std::sqrt(sum / stream_dim + kEps);
            for (std::int32_t i2 = 0; i2 < stream_dim; ++i2) {
                const std::size_t e = static_cast<std::size_t>(s * stream_dim + i2) * t + tok;
                ref[e] = x[e] * inv * gamma[s * stream_dim + i2];
            }
        }
    }

    const DeviceBuffer dx = to_device_bf16_2d(x);
    const DeviceBuffer dg = to_device_f32(gamma);
    const DeviceBuffer dout(static_cast<std::size_t>(d) * t * 2);
    const Tensor tx(dx.p, DType::BF16, {d, t});
    const Tensor tg(dg.p, DType::FP32, {d});
    Tensor tout(dout.p, DType::BF16, {d, t});
    ops::grouped_rmsnorm(tx, tg, kEps, stream_dim, tout, 0);
    cuda_synchronize();
    const std::vector<double> got = to_doubles(dout, static_cast<std::size_t>(d) * t);
    const std::string label = "grouped_rmsnorm D=" + std::to_string(d) + " stream=" +
                              std::to_string(stream_dim) + " T=" + std::to_string(t);
    return verify_reduction(label, got, ref, hc_bf16_criterion());
}

int run_hc_silu_div4(std::int32_t rows, std::int32_t t, float div, std::uint32_t seed) {
    const std::size_t n = static_cast<std::size_t>(rows) * t;
    std::vector<float> x(n);
    fill_uniform(x, seed, -8.0F, 8.0F);
    round_to_bf16(x);
    std::vector<double> ref(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = x[i] / div;
        ref[i] = v / (1.0 + std::exp(-v));
    }
    const DeviceBuffer dx = to_device_bf16_2d(x);
    const DeviceBuffer dout(n * 2);
    const Tensor tx(dx.p, DType::BF16, {rows, t});
    Tensor tout(dout.p, DType::BF16, {rows, t});
    ops::hc_silu_div4(tx, div, tout, 0);
    cuda_synchronize();
    const std::vector<double> got = to_doubles(dout, n);
    return verify_reduction("hc_silu_div4 rows=" + std::to_string(rows) + " T=" + std::to_string(t),
                            got, ref, hc_bf16_criterion());
}

int run_hc_gate_mul_mean4(std::int32_t d, std::int32_t stream_dim, std::int32_t t,
                          std::uint32_t seed) {
    const std::int32_t hc = d / stream_dim;
    const std::size_t n = static_cast<std::size_t>(d) * t;
    std::vector<float> xn(n), gate(n);
    fill_uniform(xn, seed, -4.0F, 4.0F);
    round_to_bf16(xn);
    fill_uniform(gate, seed + 1U, -6.0F, 6.0F); // pre-sigmoid up projection
    round_to_bf16(gate);

    std::vector<double> ref(static_cast<std::size_t>(stream_dim) * t);
    for (std::int32_t tok = 0; tok < t; ++tok) {
        for (std::int32_t drow = 0; drow < stream_dim; ++drow) {
            double acc = 0.0;
            for (std::int32_t s = 0; s < hc; ++s) {
                const std::size_t e = static_cast<std::size_t>(s * stream_dim + drow) * t + tok;
                acc += xn[e] / (1.0 + std::exp(-gate[e]));
            }
            ref[static_cast<std::size_t>(drow) * t + tok] = acc / hc;
        }
    }

    const DeviceBuffer dxn = to_device_bf16_2d(xn);
    const DeviceBuffer dgate = to_device_bf16_2d(gate);
    const DeviceBuffer dout(static_cast<std::size_t>(stream_dim) * t * 2);
    const Tensor txn(dxn.p, DType::BF16, {d, t});
    const Tensor tgate(dgate.p, DType::BF16, {d, t});
    Tensor tout(dout.p, DType::BF16, {stream_dim, t});
    ops::hc_gate_mul_mean4(txn, tgate, stream_dim, tout, 0);
    cuda_synchronize();
    const std::vector<double> got = to_doubles(dout, static_cast<std::size_t>(stream_dim) * t);
    const std::string label = "hc_gate_mul_mean4 D=" + std::to_string(d) + " stream=" +
                              std::to_string(stream_dim) + " T=" + std::to_string(t);
    return verify_reduction(label, got, ref, hc_bf16_criterion());
}

int run_hc_combine(std::int32_t d, std::int32_t stream_dim, std::int32_t t, std::uint32_t seed,
                   bool zero_inject) {
    const std::int32_t hc = d / stream_dim;
    const std::size_t n = static_cast<std::size_t>(d) * t;
    std::vector<float> res(n), block(static_cast<std::size_t>(stream_dim) * t), inject(hc * t);
    fill_uniform(res, seed, -4.0F, 4.0F);
    round_to_bf16(res);
    fill_uniform(block, seed + 1U, -4.0F, 4.0F);
    round_to_bf16(block);
    if (zero_inject) {
        inject.assign(hc * t, 0.0F);
    } else {
        fill_uniform(inject, seed + 2U, -4.0F, 4.0F);
        round_to_bf16(inject);
    }

    std::vector<double> ref(n);
    for (std::int32_t s = 0; s < hc; ++s) {
        for (std::int32_t tok = 0; tok < t; ++tok) {
            const double w =
                2.0 / (1.0 + std::exp(-static_cast<double>(inject[s * t + tok]) / hc));
            for (std::int32_t drow = 0; drow < stream_dim; ++drow) {
                const std::size_t e = static_cast<std::size_t>(s * stream_dim + drow) * t + tok;
                ref[e] =
                    res[e] + block[static_cast<std::size_t>(drow) * t + tok] * w;
            }
        }
    }

    const DeviceBuffer dres = to_device_bf16_2d(res);
    const DeviceBuffer dblk = to_device_bf16_2d(block);
    const DeviceBuffer dinj = to_device_bf16_2d(inject);
    const DeviceBuffer dout(n * 2);
    const Tensor tres(dres.p, DType::BF16, {d, t});
    const Tensor tblk(dblk.p, DType::BF16, {stream_dim, t});
    const Tensor tinj(dinj.p, DType::BF16, {hc, t});
    Tensor tout(dout.p, DType::BF16, {d, t});
    ops::hc_combine(tres, tblk, tinj, stream_dim, tout, 0);
    cuda_synchronize();
    const std::vector<double> got = to_doubles(dout, n);
    std::string label = "hc_combine D=" + std::to_string(d) + " stream=" + std::to_string(stream_dim) +
                        " T=" + std::to_string(t) + (zero_inject ? " (zero inject)" : "");
    return verify_reduction(label, got, ref, hc_bf16_criterion());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "no CUDA device; skipping\n";
        return 77;
    }
    int failures = 0;

    // Real qwen4exp geometry (D=10240, stream=2560, hc=4, lr=320).
    for (std::int32_t t : {1, 3, 17}) {
        failures += run_grouped_rmsnorm(10240, 2560, t, 100U + static_cast<std::uint32_t>(t));
        failures += run_hc_gate_mul_mean4(10240, 2560, t, 200U + static_cast<std::uint32_t>(t));
    }
    failures += run_hc_combine(10240, 2560, 1, 301, false);
    failures += run_hc_combine(10240, 2560, 1, 302, true); // zero inject == plain residual add
    failures += run_hc_combine(10240, 2560, 5, 303, false);
    for (std::int32_t t : {1, 3, 7}) {
        failures += run_hc_silu_div4(320, t, 4.0F, 400U + static_cast<std::uint32_t>(t));
    }

    // Synthetic small geometries.
    failures += run_grouped_rmsnorm(8, 2, 5, 501);
    failures += run_grouped_rmsnorm(16, 8, 2, 502);
    failures += run_hc_gate_mul_mean4(6, 3, 2, 503);
    failures += run_hc_gate_mul_mean4(12, 3, 1, 504); // hc=4, stream=3
    failures += run_hc_combine(4, 2, 3, 505, false);
    failures += run_hc_combine(6, 2, 1, 506, false);
    failures += run_hc_silu_div4(5, 1, 2.0F, 507);

    if (failures > 0) {
        std::cerr << failures << " hc_mix check(s) failed\n";
        return 1;
    }
    std::cout << "hc_mix: all checks passed\n";
    return 0;
}
