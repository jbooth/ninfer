// K7 `sparse_moe_512x10` qualification: the full MoE pipeline (router -> softmax -> top-10
// -> NVFP4 gate/up -> Q6 down -> shared W8 -> merge -> residual) is compared, over the exact
// stored quantized weights, against a direct FP64 oracle. The full 512-expert bank does not fit
// on the constrained device, so the op is qualified at n_experts=16 (same hidden=2560,
// inter=640, top_k=10): the routed GEMMs, routing, and merge are all exercised.
#include "ninfer/ops/sparse_moe_512x10.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;

namespace {

std::size_t align256(std::size_t v) { return (v + 255u) & ~std::size_t{255u}; }

// ---- exact weight decoders (match the kernel decode atoms) --------------------------
double e2m1_nib(int nib) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double s = (nib & 8) ? -1.0 : 1.0;
    return s * mag[nib & 7];
}
double e4m3_f(std::uint8_t b) {
    // E4M3 (no inf/nan for b<0x7E). sign b>>7, exp b>>3 &0xf, mant b&7.
    const int sign = (b >> 7) ? -1 : 1;
    const int exp = (b >> 3) & 0xf;
    const int mant = b & 7;
    if (exp == 0) { return sign * static_cast<double>(mant) * std::ldexp(1.0, -6) / 8.0; }
    return sign * std::ldexp(1.0, exp - 7) * (1.0 + mant / 8.0);
}
double nvfp4_w(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales,
               std::int64_t row, int k) {
    const auto byte = codes[row * 1280 + k / 2];
    const double e  = e2m1_nib((k & 1) ? (byte >> 4) : (byte & 0xf));
    return e * e4m3_f(scales[row * 160 + k / 16]);
}
double q6_w(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& high,
            const std::vector<std::uint8_t>& scales, std::int64_t row, int k) {
    const int g   = k / 64;
    const int sub = (k % 64) / 8;
    const int v   = k % 8;
    const std::uint32_t packed =
        codes[(row * 10 + g) * 32 + sub * 4] | (codes[(row * 10 + g) * 32 + sub * 4 + 1] << 8) |
        (codes[(row * 10 + g) * 32 + sub * 4 + 2] << 16) | (codes[(row * 10 + g) * 32 + sub * 4 + 3] << 24);
    const std::uint16_t hb_raw = high[(row * 10 + g) * 16 + sub * 2] |
                                 (high[(row * 10 + g) * 16 + sub * 2 + 1] << 8);
    const std::uint16_t hb = hb_raw ^ 0xAAAA;  // decode_eight XORs high_bits with 0xAAAA
    const std::uint16_t sb = scales[(row * 10 + g) * 2] |
                             (scales[(row * 10 + g) * 2 + 1] << 8);
    const int code4 = (v % 2 == 0) ? ((packed >> (2 * v)) & 0xf) : ((packed >> (2 * v + 14)) & 0xf);
    const int high2 = (v % 2 == 0) ? ((hb >> v) & 3) : ((hb >> (v + 7)) & 3);
    const double scale = static_cast<double>(__half2float(__ushort_as_half(sb)));
    return (static_cast<double>(code4) + static_cast<double>(high2) * 16.0 - 32.0) * scale;
}
double w8_w(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales,
            std::int64_t row, int k, int kGroups) {
    const int g    = k / 32;
    const int lane = (k % 32) / 2;
    const int par  = k % 2;
    const std::int8_t code =
        static_cast<std::int8_t>(codes[(row * kGroups + g) * 32 + lane * 2 + par]);
    const std::uint16_t sb = scales[(row * kGroups + g) * 2] |
                             (scales[(row * kGroups + g) * 2 + 1] << 8);
    return static_cast<double>(code) * static_cast<double>(__half2float(__ushort_as_half(sb)));
}
double silu_d(double g) { return g / (1.0 + std::exp(-g)); }
double sig_d(double g) { return 1.0 / (1.0 + std::exp(-g)); }

} // namespace

int main() {
    const int n_experts = 16, top_k = 10, hidden = 2560, inter = 640;
    const int H = hidden, I = inter;
    std::mt19937 rng(20260902u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::uniform_int_distribution<int> ubyte(0, 255);

    const std::size_t gu_rows = static_cast<std::size_t>(n_experts) * 2 * I;
    const std::size_t dn_rows = static_cast<std::size_t>(n_experts) * H;
    std::vector<std::uint8_t> gu_codes(gu_rows * 1280), gu_scales(gu_rows * 160);
    std::vector<std::uint8_t> dn_codes(dn_rows * 320), dn_high(dn_rows * 160), dn_scales(dn_rows * 20);
    std::vector<std::uint8_t> sg_codes(static_cast<std::size_t>(2 * I) * 80 * 32),
        sg_scales(static_cast<std::size_t>(2 * I) * 80 * 2);
    std::vector<std::uint8_t> sd_codes(static_cast<std::size_t>(H) * 20 * 32),
        sd_scales(static_cast<std::size_t>(H) * 20 * 2);
    std::vector<float> router(static_cast<std::size_t>(n_experts + 1) * H);
    for (auto& v : gu_codes) v = static_cast<std::uint8_t>(ubyte(rng));
    // NVFP4 (E4M3) gate/up scales: small values (exp in [3,7] -> [2^-4, ~1.9]) so the act stays
    // O(1e2) and the BF16 output stays well above its storage floor without inflating magnitudes.
    for (auto& v : gu_scales) {
        const int e = 3 + static_cast<int>(std::uniform_real_distribution<double>(0, 1)(rng) * 5);
        const int m = static_cast<int>(std::uniform_real_distribution<double>(0, 1)(rng) * 8);
        v = static_cast<std::uint8_t>((e << 3) | m);
    }
    for (auto& v : dn_codes) v = static_cast<std::uint8_t>(ubyte(rng));
    for (auto& v : dn_high) v = static_cast<std::uint8_t>(ubyte(rng));
    for (auto& v : sg_codes) v = static_cast<std::uint8_t>(ubyte(rng));
    for (auto& v : sd_codes) v = static_cast<std::uint8_t>(ubyte(rng));
    // Deterministic router: experts 0..9 -> +0.1, experts 10..15 -> -0.1, shared row -> 0.
    // With x=1.0 the logits are +/-0.1*H=+/-256 (gap 512 >> FP32 noise), so the top-10 is
    // unambiguously {0..9}; this isolates the routed GEMM + merge from a tight selection boundary.
    for (int e = 0; e < n_experts; ++e)
        for (int k = 0; k < H; ++k) router[static_cast<std::size_t>(e) * H + k] = (e < 10 ? 0.1f : -0.1f);
    for (int k = 0; k < H; ++k) router[static_cast<std::size_t>(n_experts) * H + k] = 0.0f;
    auto half16 = [&]() {
        const float f = 0.02f * static_cast<float>(uni(rng));
        const auto h  = __float2half(f);
        return __half_as_ushort(h);
    };
    auto fill_scale = [&](std::vector<std::uint8_t>& s) {
        for (std::size_t i = 0; i < s.size(); i += 2) {
            const std::uint16_t b = half16();
            s[i]     = b & 0xff;
            s[i + 1] = b >> 8;
        }
    };
    fill_scale(dn_scales);
    fill_scale(sg_scales);
    fill_scale(sd_scales);

    // Upload banks to device.
    std::vector<void*> devs;
    auto up = [&](const void* h, std::size_t bytes) -> void* {
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            std::fprintf(stderr, "bank upload failed\n");
            std::exit(1);
        }
        devs.push_back(d);
        return d;
    };
    auto* d_router = static_cast<const float*>(up(router.data(), router.size() * 4));
    auto* d_gu_c   = static_cast<const std::uint8_t*>(up(gu_codes.data(), gu_codes.size()));
    auto* d_gu_s   = static_cast<const std::uint8_t*>(up(gu_scales.data(), gu_scales.size()));
    auto* d_dn_c   = static_cast<const std::uint8_t*>(up(dn_codes.data(), dn_codes.size()));
    auto* d_dn_h   = static_cast<const std::uint8_t*>(up(dn_high.data(), dn_high.size()));
    auto* d_dn_s   = static_cast<const std::uint8_t*>(up(dn_scales.data(), dn_scales.size()));
    auto* d_sg_c   = static_cast<const std::uint8_t*>(up(sg_codes.data(), sg_codes.size()));
    auto* d_sg_s   = static_cast<const std::uint8_t*>(up(sg_scales.data(), sg_scales.size()));
    auto* d_sd_c   = static_cast<const std::uint8_t*>(up(sd_codes.data(), sd_codes.size()));
    auto* d_sd_s   = static_cast<const std::uint8_t*>(up(sd_scales.data(), sd_scales.size()));

    const SparseMoe512x10Geometry geo{n_experts, top_k, H, I};
    const SparseMoe512x10Weights w{d_router, 1.0f, d_gu_c, d_gu_s, d_dn_c, d_dn_h, d_dn_s,
                                   d_sg_c, d_sg_s, d_sd_c, d_sd_s};

    int failures = 0;
    for (const int T : {1, 2, 4}) {
    // Constant small x: keeps the deterministic router's top-10 unambiguous (logit[e<10] = 0.1*c*H)
    // while keeping the act/output magnitude realistic (O(1e2-1e3)) so the BF16 criterion is tight.
        std::vector<float> x_host(static_cast<std::size_t>(H) * T, 0.05f),
            res_host(static_cast<std::size_t>(H) * T, 0.0f);
        DeviceBuffer dx = to_device_bf16(x_host), dest = to_device_bf16(res_host);
        std::vector<float> xbf = x_host, resbf = res_host;
        round_to_bf16(xbf);
        round_to_bf16(resbf);
        std::vector<double> x64(xbf.begin(), xbf.end()), res64(resbf.begin(), resbf.end());
        Tensor xt{dx.p, DType::BF16, {H, T}};
        Tensor dt{dest.p, DType::BF16, {H, T}};
        WorkspaceArena ws(align256(sparse_moe_512x10_workspace_capacity_bytes(geo, T)));
        sparse_moe_512x10(xt, geo, w, SparseMoe512x10Epilogue::AddResidual, dt, ws, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "op failed T=%d: %s\n", T, cudaGetErrorString(cudaGetLastError()));
            return 1;
        }

        // ---- FP64 oracle ----
        std::vector<double> logit(static_cast<std::size_t>(n_experts + 1) * T);
        for (int t = 0; t < T; ++t)
            for (int e = 0; e <= n_experts; ++e) {
                double a = 0;
                for (int k = 0; k < H; ++k) a += static_cast<double>(router[e * H + k]) * x64[k * T + t];
                logit[e * T + t] = a;
            }
        std::vector<double> topk_ids64(top_k * T, -1.0), topk_w64(top_k * T, 0.0), shared_logit64(T);
        for (int t = 0; t < T; ++t) {
            double mx = -1e300, sum = 0;
            for (int e = 0; e < n_experts; ++e) mx = std::fmax(mx, logit[e * T + t]);
            std::vector<double> p(n_experts);
            for (int e = 0; e < n_experts; ++e) { p[e] = std::exp(logit[e * T + t] - mx); sum += p[e]; }
            for (int e = 0; e < n_experts; ++e) p[e] /= sum;
            shared_logit64[t] = logit[n_experts * T + t];
            std::vector<int> ids(n_experts);
            for (int e = 0; e < n_experts; ++e) ids[e] = e;
            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                if (p[a] != p[b]) return p[a] > p[b];
                return a < b;
            });
            double wsum = 0;
            for (int i = 0; i < top_k; ++i) { topk_ids64[i * T + t] = ids[i]; wsum += p[ids[i]]; }
            const double denom = std::fmax(wsum, 6.1035e-5);
            for (int i = 0; i < top_k; ++i) topk_w64[i * T + t] = p[ids[i]] / denom;
        }
        std::vector<double> sact(static_cast<std::size_t>(I) * T), sy(static_cast<std::size_t>(H) * T);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < I; ++j) {
                double g = 0, u = 0;
                for (int k = 0; k < H; ++k) {
                    g += w8_w(sg_codes, sg_scales, j, k, 80) * x64[k * T + t];
                    u += w8_w(sg_codes, sg_scales, I + j, k, 80) * x64[k * T + t];
                }
                sact[j * T + t] = silu_d(g) * u;
            }
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < H; ++i) {
                double a = 0;
                for (int k = 0; k < I; ++k) a += w8_w(sd_codes, sd_scales, i, k, 20) * sact[k * T + t];
                sy[i * T + t] = a;
            }
        std::vector<double> out(static_cast<std::size_t>(H) * T, 0.0);
        for (int t = 0; t < T; ++t)
            for (int el = 0; el < top_k; ++el) {
                const int e = static_cast<int>(topk_ids64[el * T + t]);
                if (e < 0) { continue; }
                std::vector<double> act(I);
                for (int j = 0; j < I; ++j) {
                    double g = 0, u = 0;
                    for (int k = 0; k < H; ++k) {
                        g += nvfp4_w(gu_codes, gu_scales, static_cast<std::int64_t>(e) * 2 * I + j, k) * x64[k * T + t];
                        u += nvfp4_w(gu_codes, gu_scales, static_cast<std::int64_t>(e) * 2 * I + I + j, k) * x64[k * T + t];
                    }
                    act[j] = silu_d(g) * u;
                }
                const double wgt = topk_w64[el * T + t];
                for (int i = 0; i < H; ++i) {
                    double a = 0;
                    for (int k = 0; k < I; ++k)
                        a += q6_w(dn_codes, dn_high, dn_scales, static_cast<std::int64_t>(e) * H + i, k) * act[k];
                    out[i * T + t] += wgt * a;
                }
            }
        std::vector<double> ref(static_cast<std::size_t>(H) * T);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < H; ++i)
                ref[i * T + t] = res64[i * T + t] + out[i * T + t] + sig_d(shared_logit64[t]) * sy[i * T + t];

        const std::vector<double> got = from_device_bf16(dest.p, static_cast<std::size_t>(H) * T);
        failures += verify_reduction("K7 MoE n=16 T=" + std::to_string(T), got, ref,
                                     ReductionCriterion{5.0e-3, 5.0e-3, 4.2e-3});
    }
    for (auto* d : devs) { cudaFree(d); }
    std::printf(failures == 0 ? "sparse_moe_512x10: all checks passed\n" : "sparse_moe_512x10: %d FAILED\n",
                failures);
    return failures == 0 ? 0 : 1;
}
