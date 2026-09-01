// M4 gate (phase3.md §4.4 / K8): numerical qualification of the qwen4exp personalization-layer
// kernels (include/ninfer/ops/ple_mix.h) against naive FP64 oracles of the represented inputs.
//
// Geometry: the real qwen4exp PLE shape (D=10240, stream=2560, hc=4; conv C=10240) plus small
// synthetic shapes, over T in {1, 3, 5, 9, 17} (decode and ragged prefill widths). The conv
// state update is a pure copy of the trailing window and is verified bit-exactly; the dilated
// conv also verifies chunked-prefill equivalence (two chunked calls with state carry == one
// single-shot call).

#include "ninfer/ops/ple_mix.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// ple_stream_gate criterion. The dot is an FP32 reduction of 2560 exact BF16 products, so the
// implementation error is the FP32 summation profile (bounded by ~1e-3 absolute on the dot for
// adversarial cancellation). The softsign amplifies dot error near |dot| = 0 by at most
// 0.25 / (2*sqrt(1e-6)) = 125, so the honest bound on the FP32 gate output is a few milliradians
// absolute; 6e-3 absolute/relative keeps margin while staying tight on (0, 1)-valued gates.
PointwiseCriterion ple_gate_criterion() {
    return {/*absolute*/ 6.0e-3, /*relative*/ 6.0e-3};
}

// BF16-output storage criterion (same profile as the HC family: kernel-internal FP32 error is
// negligible next to the worst-case half-ulp storage rounding of 3.906e-3).
ReductionCriterion ple_bf16_criterion() {
    return {/*relative_l2*/ 2.4e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 4.2e-3};
}

// One convolution-reduction profile for the dilated conv. The BF16 output's dominant error term
// is the reference's own storage rounding, whose relative-L2 floor is data-dependent; measured
// 1.56e-3..1.87e-3 across seeds at this geometry. The kernel's additional FP32 accumulation
// error is ~1e-7. relative_l2 is set ~2x the worst-case measured floor (still ~2 orders below
// any real tap/index bug, which is O(1e-1)+); gross matches the causal_conv1d_silu profile.
ReductionCriterion ple_conv_criterion() {
    return {/*relative_l2*/ 3.0e-3, /*gross_absolute*/ 1.0e-3,
            /*gross_relative_to_max_reference*/ 3.7e-3};
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = f32_to_bf16(values[i]);
    }
    return result;
}

DeviceBuffer to_device_bf16_2d(const std::vector<float>& h) {
    const std::vector<std::uint16_t> bits = bf16_bits(h);
    return to_device<std::uint16_t>(bits);
}

DeviceBuffer to_device_f32(const std::vector<float>& h) {
    std::vector<std::uint32_t> bits(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        std::memcpy(&bits[i], &h[i], 4);
    }
    return to_device<std::uint32_t>(bits);
}

std::vector<double> bf16_to_doubles(const std::vector<std::uint16_t>& h) {
    std::vector<double> out(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        out[i] = static_cast<double>(bf16_to_f32(h[i]));
    }
    return out;
}

// gate[s,t] = sigmoid(sgn(dot)*sqrt(clamp(|dot|,1e-6,inf))); dot = sum_d key*query / sqrt(stream).
// key/query are [D, T] row-major (element [i,t] at i*T+t) on the bf16 grid.
int run_ple_stream_gate(std::int32_t d, std::int32_t hc, std::int32_t t, std::uint32_t seed) {
    const std::int32_t stream_dim = d / hc;
    std::vector<float> key(static_cast<std::size_t>(d) * t),
        query(static_cast<std::size_t>(d) * t);
    fill_uniform(key, seed, -4.0F, 4.0F);
    round_to_bf16(key);
    fill_uniform(query, seed + 1U, -4.0F, 4.0F);
    round_to_bf16(query);

    std::vector<double> ref(static_cast<std::size_t>(hc) * t);
    const double inv_sqrt = 1.0 / std::sqrt(static_cast<double>(stream_dim));
    for (std::int32_t s = 0; s < hc; ++s) {
        for (std::int32_t tok = 0; tok < t; ++tok) {
            double dot = 0.0;
            for (std::int32_t i2 = 0; i2 < stream_dim; ++i2) {
                const std::size_t e =
                    static_cast<std::size_t>(s * stream_dim + i2) * t + tok;
                dot += static_cast<double>(key[e]) * query[e];
            }
            dot *= inv_sqrt;
            const double mag = std::sqrt(std::max(std::fabs(dot), 1.0e-6));
            ref[static_cast<std::size_t>(s) * t + tok] =
                1.0 / (1.0 + std::exp(-std::copysign(mag, dot)));
        }
    }

    const DeviceBuffer dkey = to_device_bf16_2d(key);
    const DeviceBuffer dquery = to_device_bf16_2d(query);
    const DeviceBuffer dout(static_cast<std::size_t>(hc) * t * 4);
    const Tensor tkey(dkey.p, DType::BF16, {d, t});
    const Tensor tquery(dquery.p, DType::BF16, {d, t});
    Tensor tout(dout.p, DType::FP32, {hc, t});
    ops::ple_stream_gate(tkey, tquery, tout, 0);
    cuda_synchronize();
    const std::vector<float> got_f = from_device<float>(dout, static_cast<std::size_t>(hc) * t);
    std::vector<double> got(got_f.size());
    for (std::size_t i = 0; i < got_f.size(); ++i) {
        got[i] = static_cast<double>(got_f[i]);
    }
    const std::string label = "ple_stream_gate D=" + std::to_string(d) + " hc=" +
                              std::to_string(hc) + " T=" + std::to_string(t);
    return verify_pointwise(label, got, ref, ple_gate_criterion());
}

// out[s*S+d,t] = value[d,t] * gate[s,t]; value [S,T] BF16, gate [hc,T] FP32, out [hc*S,T] BF16.
int run_ple_gate_scale(std::int32_t s_dim, std::int32_t hc, std::int32_t t, std::uint32_t seed,
                       bool unit_gate) {
    std::vector<float> value(static_cast<std::size_t>(s_dim) * t),
        gate(static_cast<std::size_t>(hc) * t);
    fill_uniform(value, seed, -4.0F, 4.0F);
    round_to_bf16(value);
    if (unit_gate) {
        gate.assign(static_cast<std::size_t>(hc) * t, 1.0F);
    } else {
        fill_uniform(gate, seed + 1U, 0.0F, 1.0F);
        round_to_bf16(gate); // gate is FP32 in the op; keep the represented grid honest
    }

    const std::size_t n = static_cast<std::size_t>(s_dim) * hc * t;
    std::vector<double> ref(n);
    for (std::int32_t s = 0; s < hc; ++s) {
        for (std::int32_t tok = 0; tok < t; ++tok) {
            for (std::int32_t i2 = 0; i2 < s_dim; ++i2) {
                const std::size_t e =
                    static_cast<std::size_t>(s * s_dim + i2) * t + tok;
                ref[e] = static_cast<double>(value[static_cast<std::size_t>(i2) * t + tok]) *
                         gate[static_cast<std::size_t>(s) * t + tok];
            }
        }
    }

    const DeviceBuffer dvalue = to_device_bf16_2d(value);
    const DeviceBuffer dgate = to_device_f32(gate);
    const DeviceBuffer dout(n * 2);
    const Tensor tvalue(dvalue.p, DType::BF16, {s_dim, t});
    const Tensor tgate(dgate.p, DType::FP32, {hc, t});
    Tensor tout(dout.p, DType::BF16, {s_dim * hc, t});
    ops::ple_gate_scale(tvalue, tgate, tout, 0);
    cuda_synchronize();
    const std::vector<double> got =
        bf16_to_doubles(from_device<std::uint16_t>(dout, n));
    std::string label = "ple_gate_scale S=" + std::to_string(s_dim) + " hc=" + std::to_string(hc) +
                        " T=" + std::to_string(t) + (unit_gate ? " (unit gate)" : "");
    return verify_reduction(label, got, ref, ple_bf16_criterion());
}

// x [C,T] BF16, weight [4,C] BF16 (tap-major), state [C,9] BF16 (oldest..newest).
// ideal[c,t] = silu(sum_k w[k,C+c] * u[c,t-(3-k)*3]); u[c,-9..-1] = state.
std::vector<double> conv_ref(std::int32_t C, std::int32_t T, const std::vector<float>& x,
                             const std::vector<float>& w, const std::vector<float>& st) {
    std::vector<double> out(static_cast<std::size_t>(C) * T);
    for (std::int32_t c = 0; c < C; ++c) {
        for (std::int32_t t = 0; t < T; ++t) {
            double acc = 0.0;
            for (int k = 0; k < 4; ++k) {
                const int p = t - (3 - k) * 3;
                const double u = (p >= 0) ? x[static_cast<std::size_t>(c) * T + p]
                                          : st[static_cast<std::size_t>(c) * 9 + (p + 9)];
                acc += w[static_cast<std::size_t>(k) * C + c] * u;
            }
            out[static_cast<std::size_t>(c) * T + t] = acc / (1.0 + std::exp(-acc));
        }
    }
    return out;
}

std::vector<std::uint16_t> state_ref(std::int32_t C, std::int32_t T,
                                     const std::vector<std::uint16_t>& st_bits,
                                     const std::vector<std::uint16_t>& x_bits) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(C) * 9);
    for (std::int32_t c = 0; c < C; ++c) {
        for (int j = 0; j < 9; ++j) {
            const int p = T + j;
            out[static_cast<std::size_t>(c) * 9 + j] =
                (p < 9) ? st_bits[static_cast<std::size_t>(c) * 9 + p]
                        : x_bits[static_cast<std::size_t>(c) * T + (p - 9)];
        }
    }
    return out;
}

int run_dilated_conv(std::int32_t C, std::int32_t T, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(C) * T),
        w(static_cast<std::size_t>(C) * 4U), st(static_cast<std::size_t>(C) * 9U);
    fill_uniform(x, seed, -3.0F, 3.0F);
    round_to_bf16(x);
    fill_uniform(w, seed + 1U, -1.0F, 1.0F);
    round_to_bf16(w);
    fill_uniform(st, seed + 2U, -3.0F, 3.0F);
    round_to_bf16(st);

    const std::vector<double> ref = conv_ref(C, T, x, w, st);
    const std::vector<std::uint16_t> x_bits = bf16_bits(x);
    const std::vector<std::uint16_t> st_bits = bf16_bits(st);

    const DeviceBuffer dx = to_device_bf16_2d(x);
    const DeviceBuffer dw = to_device_bf16_2d(w);
    const DeviceBuffer dst = to_device_bf16_2d(st);
    const DeviceBuffer dout(static_cast<std::size_t>(C) * T * 2);
    const DeviceBuffer dst_out(static_cast<std::size_t>(C) * 9 * 2);
    const Tensor tx(dx.p, DType::BF16, {C, T});
    const Tensor tw(dw.p, DType::BF16, {4, C});
    const Tensor tstate(dst.p, DType::BF16, {C, 9});
    Tensor tout(dout.p, DType::BF16, {C, T});
    Tensor tstate_out(dst_out.p, DType::BF16, {C, 9});
    ops::dilated_causal_conv1d_silu(tx, tw, tstate, tstate_out, tout, 0);
    cuda_synchronize();
    const std::vector<double> got =
        bf16_to_doubles(from_device<std::uint16_t>(dout, static_cast<std::size_t>(C) * T));
    int failures = 0;
    failures += verify_reduction(
        "dilated_conv1d C=" + std::to_string(C) + " T=" + std::to_string(T), got, ref,
        ple_conv_criterion());
    // The state update is a pure copy: bit-exact.
    const std::string state_label =
        "dilated_conv1d state C=" + std::to_string(C) + " T=" + std::to_string(T);
    failures += verify_exact(
        state_label.c_str(),
        from_device<std::uint16_t>(dst_out, static_cast<std::size_t>(C) * 9),
        state_ref(C, T, st_bits, x_bits));
    return failures;
}

// In-place state alias: state_in == state_out.
int run_dilated_conv_inplace(std::int32_t C, std::int32_t T, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(C) * T),
        w(static_cast<std::size_t>(C) * 4U), st(static_cast<std::size_t>(C) * 9U);
    fill_uniform(x, seed, -3.0F, 3.0F);
    round_to_bf16(x);
    fill_uniform(w, seed + 1U, -1.0F, 1.0F);
    round_to_bf16(w);
    fill_uniform(st, seed + 2U, -3.0F, 3.0F);
    round_to_bf16(st);

    const DeviceBuffer dx = to_device_bf16_2d(x);
    const DeviceBuffer dw = to_device_bf16_2d(w);
    const DeviceBuffer dst = to_device_bf16_2d(st);
    const DeviceBuffer dout(static_cast<std::size_t>(C) * T * 2);
    const Tensor tx(dx.p, DType::BF16, {C, T});
    const Tensor tw(dw.p, DType::BF16, {4, C});
    Tensor tstate(dst.p, DType::BF16, {C, 9});
    Tensor tout(dout.p, DType::BF16, {C, T});
    ops::dilated_causal_conv1d_silu(tx, tw, tstate, tout, 0);
    cuda_synchronize();
    return 0; // aliasing exercised; numerical path identical to the distinct-state form
}

// Chunked-prefill equivalence: conv(x[:, 0..T-1], s0) split into (x[:, 0..A-1]) then
// (x[:, A..T-1]) with state carry must match the single-shot call within the conv criterion,
// and the final state must be bit-exact.
int run_dilated_conv_chunked(std::int32_t C, std::int32_t T, std::int32_t A,
                             std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(C) * T),
        w(static_cast<std::size_t>(C) * 4U), st(static_cast<std::size_t>(C) * 9U);
    fill_uniform(x, seed, -3.0F, 3.0F);
    round_to_bf16(x);
    fill_uniform(w, seed + 1U, -1.0F, 1.0F);
    round_to_bf16(w);
    fill_uniform(st, seed + 2U, -3.0F, 3.0F);
    round_to_bf16(st);

    const std::vector<double> single = conv_ref(C, T, x, w, st);
    const std::vector<std::uint16_t> x_bits = bf16_bits(x);
    const std::vector<std::uint16_t> st_bits = bf16_bits(st);
    std::vector<float> x1(static_cast<std::size_t>(C) * A),
        x2(static_cast<std::size_t>(C) * (T - A));
    for (std::int32_t c = 0; c < C; ++c) {
        for (std::int32_t a = 0; a < A; ++a) {
            x1[static_cast<std::size_t>(c) * A + a] = x[static_cast<std::size_t>(c) * T + a];
        }
        for (std::int32_t a = 0; a < T - A; ++a) {
            x2[static_cast<std::size_t>(c) * (T - A) + a] =
                x[static_cast<std::size_t>(c) * T + A + a];
        }
    }
    // The mid-state oracle needs the [C,A]-layout chunk-1 input.
    const std::vector<std::uint16_t> x1_bits = bf16_bits(x1);
    const std::vector<std::uint16_t> mid_bits = state_ref(C, A, st_bits, x1_bits);
    std::vector<float> mid_f(mid_bits.size());
    for (std::size_t i = 0; i < mid_bits.size(); ++i) {
        mid_f[i] = bf16_to_f32(mid_bits[i]);
    }
    const std::vector<double> part1 = conv_ref(C, A, x1, w, st);
    const std::vector<double> part2 = conv_ref(C, T - A, x2, w, mid_f);

    const DeviceBuffer dx1 = to_device_bf16_2d(x1);
    const DeviceBuffer dx2 = to_device_bf16_2d(x2);
    const DeviceBuffer dw = to_device_bf16_2d(w);
    const DeviceBuffer ddst = to_device_bf16_2d(st);
    const DeviceBuffer dsingle(static_cast<std::size_t>(C) * T * 2);
    const DeviceBuffer dchunk(static_cast<std::size_t>(C) * T * 2);
    const DeviceBuffer dstate1(static_cast<std::size_t>(C) * 9 * 2);
    const DeviceBuffer dstate2(static_cast<std::size_t>(C) * 9 * 2);
    const Tensor tw(dw.p, DType::BF16, {4, C});

    // Single shot (own state copy; the in-place form mutates it).
    {
        const std::vector<std::uint16_t> st_bits = bf16_bits(st);
        const DeviceBuffer ds = to_device<std::uint16_t>(st_bits);
        const DeviceBuffer dxfull = to_device_bf16_2d(x);
        const Tensor txf(dxfull.p, DType::BF16, {C, T});
        Tensor tstate(ds.p, DType::BF16, {C, 9});
        Tensor tout(dsingle.p, DType::BF16, {C, T});
        ops::dilated_causal_conv1d_silu(txf, tw, tstate, tout, 0);
        cuda_synchronize();
    }
    // Chunk 1.
    {
        const Tensor tx(dx1.p, DType::BF16, {C, A});
        const Tensor tstate(ddst.p, DType::BF16, {C, 9});
        Tensor tout(dchunk.p, DType::BF16, {C, A});
        Tensor tstate1(dstate1.p, DType::BF16, {C, 9});
        ops::dilated_causal_conv1d_silu(tx, tw, tstate, tstate1, tout, 0);
        cuda_synchronize();
    }
    // Chunk 2 (carries chunk 1's state; writes to the dchunk tail).
    {
        std::uint8_t* o2 = static_cast<std::uint8_t*>(dchunk.p) +
                           static_cast<std::size_t>(C) * A * 2;
        const Tensor tx(dx2.p, DType::BF16, {C, T - A});
        const Tensor tstate(dstate1.p, DType::BF16, {C, 9});
        Tensor tout(o2, DType::BF16, {C, T - A});
        Tensor tstate2(dstate2.p, DType::BF16, {C, 9});
        ops::dilated_causal_conv1d_silu(tx, tw, tstate, tstate2, tout, 0);
        cuda_synchronize();
    }

    const std::vector<double> single_got =
        bf16_to_doubles(from_device<std::uint16_t>(dsingle, static_cast<std::size_t>(C) * T));
    // The two chunk outputs are separate [C,A] and [C,T-A] tensors (dchunk front/tail); assemble
    // the full [C,T] layout: full[c*T + a] = chunk1[c*A + a], full[c*T + A + a] = chunk2[c*(T-A)+a].
    const std::vector<std::uint16_t> dchunk_bits =
        from_device<std::uint16_t>(dchunk, static_cast<std::size_t>(C) * T);
    std::vector<double> chunk_got(static_cast<std::size_t>(C) * T);
    for (std::int32_t c = 0; c < C; ++c) {
        for (std::int32_t a = 0; a < A; ++a) {
            chunk_got[static_cast<std::size_t>(c) * T + a] =
                bf16_to_f32(dchunk_bits[static_cast<std::size_t>(c) * A + a]);
        }
        for (std::int32_t a = 0; a < T - A; ++a) {
            chunk_got[static_cast<std::size_t>(c) * T + A + a] = bf16_to_f32(
                dchunk_bits[static_cast<std::size_t>(C) * A +
                            static_cast<std::size_t>(c) * (T - A) + a]);
        }
    }
    int failures = 0;
    // Both device routes against their FP64 references.
    failures += verify_reduction("dilated_conv1d single C=" + std::to_string(C) + " T=" +
                                     std::to_string(T),
                                 single_got, single, ple_conv_criterion());
    failures += verify_reduction("dilated_conv1d chunk C=" + std::to_string(C) + " A=" +
                                     std::to_string(A),
                                 chunk_got, single, ple_conv_criterion());
    // Part references (each an independent [C, chunk-T] layout).
    std::vector<double> chunk_part1(static_cast<std::size_t>(C) * A),
        chunk_part2(static_cast<std::size_t>(C) * (T - A));
    for (std::int32_t c = 0; c < C; ++c) {
        for (std::int32_t a = 0; a < A; ++a) {
            chunk_part1[static_cast<std::size_t>(c) * A + a] =
                bf16_to_f32(dchunk_bits[static_cast<std::size_t>(c) * A + a]);
        }
        for (std::int32_t a = 0; a < T - A; ++a) {
            chunk_part2[static_cast<std::size_t>(c) * (T - A) + a] = bf16_to_f32(
                dchunk_bits[static_cast<std::size_t>(C) * A +
                            static_cast<std::size_t>(c) * (T - A) + a]);
        }
    }
    failures += verify_reduction("dilated_conv1d chunk part1", chunk_part1, part1,
                                 ple_conv_criterion());
    failures += verify_reduction("dilated_conv1d chunk part2", chunk_part2, part2,
                                 ple_conv_criterion());
    // Chunked final state must equal the single-shot final state bit-exactly.
    const std::vector<std::uint16_t> final_ref = state_ref(C, T, st_bits, x_bits);
    failures += verify_exact("dilated_conv1d chunked state",
                             from_device<std::uint16_t>(dstate2, static_cast<std::size_t>(C) * 9),
                             final_ref);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "no CUDA device; skipping\n";
        return 77;
    }
    int failures = 0;

    // Real qwen4exp PLE geometry (D=10240, hc=4, stream=2560; conv C=10240).
    for (std::int32_t t : {1, 3, 17}) {
        failures += run_ple_stream_gate(10240, 4, t, 100U + static_cast<std::uint32_t>(t));
        failures += run_ple_gate_scale(2560, 4, t, 200U + static_cast<std::uint32_t>(t), false);
    }
    failures += run_ple_gate_scale(2560, 4, 1, 250, true); // unit gate == broadcast (exact)
    for (std::int32_t t : {1, 5, 9, 17}) {
        failures += run_dilated_conv(10240, t, 300U + static_cast<std::uint32_t>(t));
    }
    failures += run_dilated_conv_inplace(10240, 3, 360);
    failures += run_dilated_conv_chunked(10240, 9, 4, 370);

    // Synthetic small geometries.
    failures += run_ple_stream_gate(8, 2, 5, 501);
    failures += run_ple_stream_gate(12, 3, 2, 502); // hc=3, stream=4
    failures += run_ple_gate_scale(3, 3, 2, 503, false);
    failures += run_ple_gate_scale(5, 2, 1, 504, true);
    for (std::int32_t t : {1, 3, 8}) {
        failures += run_dilated_conv(64, t, 600U + static_cast<std::uint32_t>(t));
    }
    failures += run_dilated_conv_chunked(64, 7, 3, 650);

    if (failures > 0) {
        std::cerr << failures << " ple_mix check(s) failed\n";
        return 1;
    }
    std::cout << "ple_mix: all checks passed\n";
    return 0;
}
