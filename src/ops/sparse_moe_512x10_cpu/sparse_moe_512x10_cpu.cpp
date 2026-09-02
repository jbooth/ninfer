// src/ops/sparse_moe_512x10_cpu/sparse_moe_512x10_cpu.cpp
//
// Host CPU implementation of the Qwen4Exp 512-expert top-10 routed MoE (jbq4). W4A8/W8A8
// integer GEMM: the input activation is quantized to int8 once per token, each expert's
// swiglu intermediate is quantized to int8 before its down GEMM, and the per-64-group integer
// dot is exact (int32) before the FP16 weight scale and the token activation scale are
// applied in FP32. Q4 (packed nibble) and Q8 (raw i8) decode to the same signed code, so one
// GEMV core serves both.
//
// The top-10 selection is per-token, so each (token, rank) job decodes exactly one token's
// activation and runs that expert's gate/up -> swiglu -> down GEMV. Jobs are independent and
// use a fixed k-sequential association order, so the output is bit-deterministic for a fixed
// (input, bank) regardless of thread partitioning.
//
// The integer GEMV uses AVX512-VNNI (vpdpbusd) in a 16-row-parallel, 64-wide microkernel:
// 16 output-neuron rows are processed as 16 dpbusd lanes per 64-element k-group. Weights
// are loaded as 16 strided contiguous rows (row-major, K-contiguous), unpacked to u8
// (code+128) encoding, interleaved into 16 window registers (lane l = row l, 4 bytes =
// 4 consecutive K-codes), and dot-producted against the int8 activation broadcast.
// The per-group int32 result carries a 128*S_g bias (S_g = sum of 64 activation bytes)
// that is subtracted before the FP16 scale multiplication. Q4 (packed nibble) and Q8
// (raw i8) decode to the same u8 encoding, so one GEMV core serves both.
//
// The FP accumulation order matches the scalar reference: per-group f16_scale * int32_dot
// is accumulated into an f32 sum, then multiplied by the per-token activation scale at
// the end. This guarantees bit-exactness with the scalar oracle.
#include "ninfer/ops/sparse_moe_512x10_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#pragma GCC target("avx512f,avx512vnni,avx512vl,avx512dq")
#include <immintrin.h>

namespace {

// ---- BF16 / FP16 bit conversions ------------------------------------------------------
inline float bf16_to_f32(std::int16_t v) {
    std::uint32_t bits = static_cast<std::uint32_t>(static_cast<std::uint16_t>(v)) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline std::int16_t f32_to_bf16(float f) {  // round-to-nearest-even
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    std::uint32_t r = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    return static_cast<std::int16_t>(static_cast<std::uint16_t>(r >> 16));
}
inline float f16_to_f32(std::uint16_t h) {
    const int s = (h >> 15) & 1;
    const int e = (h >> 10) & 0x1F;
    const int m = h & 0x3FF;
    float v;
    if (e == 0) {
        v = (m / 1024.0f) * 0x0p-14f;  // subnormal
    } else if (e == 0x1F) {
        v = (m == 0) ? std::numeric_limits<float>::infinity() : std::nanf("");
    } else {
        v = (1.0f + m / 1024.0f) * std::ldexp(1.0f, e - 15);
    }
    return s ? -v : v;
}
inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// VNNI GEMV: out[r*T + t] = ascale[t] * sum_g f16(scale[r*groups+g]) * (sum_{k in g} code(r,k) * aq[t,k]).
//
// 16-row parallel microkernel, 64-wide k-groups. Row-major storage: K is the contiguous
// axis, so each output-neuron row's 64-code slice is one contiguous aligned load
// (32 B for Q4, 64 B for Q8). The 16 rows are strided by row_stride.
//
// Per 256-wide K-window per 16-row block (the window holds up to 4 x 64-group microkernel
// runs; a shorter final window covers K % 256 != 0):
//   1. Decode 16 rows x 256 codes to u8 (code+128) in the 4096 B stack buffer:
//      Q8 pulls 256 i8 in once (4 x 512-bit loads), Q4 pulls 128 packed bytes in once
//      (4 x 256-bit loads).
//   2. Interleave into the window's 64 x 4-byte WIN[w] slots: WIN[w] lane l = u8w[l][4w..4w+3].
//   3. Pull the window's 256 activation bytes and the 4 groups' per-row FP16 scales in once.
//   4. Run the 64-wide microkernel over the L1-resident buffer once per 64-group:
//      16 x vpdpbusd (u8 WIN x i8 activation broadcast) -> 16 int32 partial dots,
//      subtract 128*S_g (S_g = sum of the group's 64 activation bytes) for the u8 bias,
//      multiply by the per-row FP16 scale, accumulate into tmp16.
// After all windows: out = ascale * tmp16.
//
// Row alignment: row_stride is always a multiple of the 64-code slice (32 B Q4 / 64 B Q8)
// because K % 64 == 0, so each row's slice starts at a 32/64 B-aligned offset within the
// 256 B-aligned base plane.
void int8_gemv(ninfer::ops::MoeCode codec, const std::uint8_t* base, const std::uint16_t* scales,
               int n_rows, int k, const std::int8_t* aq, const float* ascale, int T, float* out) {
    const int groups = k / 64;
    const int row_stride = (codec == ninfer::ops::MoeCode::Q4G64) ? (k / 2) : k;
    const int group_bytes = (codec == ninfer::ops::MoeCode::Q4G64) ? 32 : 64;

    for (int t = 0; t < T; ++t) {
        const std::int8_t* aq_t = aq + static_cast<std::size_t>(t) * k;
        const float asc = ascale[t];

        for (int row0 = 0; row0 < n_rows; row0 += 16) {
            const int rows_this = std::min(16, n_rows - row0);
            float tmp16[16] = {};

            // 256-wide K window: decode one 4096 B buffer (16 rows x 256 codes), pull the
            // window's 256 activation bytes and per-row scales in once, then run the
            // 64-wide microkernel over the L1-resident buffer once per 64-group.
            for (int g0 = 0; g0 < groups; g0 += 4) {
                const int nw = std::min(4, groups - g0);   // 64-groups in this window
                const int W  = nw * 64;                    // K-codes in this window

                // ---- 1. Decode 16 rows x W codes to u8 (code+128) ----
                std::uint8_t u8w[16][256];
                if (codec == ninfer::ops::MoeCode::Q4G64) {
                    const __m256i mask_lo = _mm256_set1_epi8(0x0F);
                    const __m256i mask_hi = _mm256_set1_epi8(static_cast<int8_t>(0xF0));
                    for (int l = 0; l < rows_this; ++l) {
                        const std::uint8_t* src =
                            base + static_cast<std::size_t>(row0 + l) * row_stride + g0 * 32;
                        for (int c = 0; c < nw; ++c) {
                            // 32 packed bytes -> 32 low + 32 high nibbles
                            std::uint8_t lo[32], hi[32];
                            const __m256i v =
                                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c * 32));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(lo),
                                                _mm256_and_si256(v, mask_lo));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(hi),
                                                _mm256_srli_epi32(_mm256_and_si256(v, mask_hi), 4));
                            // interleave + bias: (nib^8)+120 = signed_code+128
                            std::uint8_t* dst = u8w[l] + c * 64;
                            for (int i = 0; i < 32; ++i) {
                                dst[2*i]   = static_cast<std::uint8_t>((lo[i] ^ 0x08) + 120);
                                dst[2*i+1] = static_cast<std::uint8_t>((hi[i] ^ 0x08) + 120);
                            }
                        }
                    }
                } else {
                    const __m512i xor_u8 = _mm512_set1_epi8(0x80);  // int8 + 128
                    for (int l = 0; l < rows_this; ++l) {
                        const std::uint8_t* src =
                            base + static_cast<std::size_t>(row0 + l) * row_stride + g0 * 64;
                        for (int c = 0; c < nw; ++c) {
                            const __m512i row =
                                _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + c * 64));
                            _mm512_storeu_si512(reinterpret_cast<__m512i*>(u8w[l] + c * 64),
                                                _mm512_xor_si512(row, xor_u8));
                        }
                    }
                }
                // Pad missing rows: u8=128 means code=0, dpbusd contribution = 0.
                for (int l = rows_this; l < 16; ++l) {
                    std::memset(&u8w[l], 128, 256);
                }

                // ---- 2. Interleave: u8w[l][p] -> win_data[w][l] (4-byte windows) ----
                std::int32_t win_data[64][16];
                for (int l = 0; l < 16; ++l) {
                    for (int w = 0; w < nw * 16; ++w) {
                        std::memcpy(&win_data[w][l], &u8w[l][4 * w], 4);
                    }
                }

                // ---- 3. Activation: the window's W bytes in once (L1-resident); per-64 sums ----
                std::int8_t act_win[256];
                std::memcpy(act_win, aq_t + g0 * 64, W);
                int Sg[4] = {};
                for (int c = 0; c < nw; ++c) {
                    const std::int8_t* p = act_win + c * 64;
                    for (int i = 0; i < 64; ++i) Sg[c] += p[i];
                }

                // ---- 4. Scales: the window's nw FP16 values per row in once ----
                float sf[4][16] = {};
                for (int l = 0; l < rows_this; ++l) {
                    for (int c = 0; c < nw; ++c) {
                        sf[c][l] =
                            f16_to_f32(scales[static_cast<std::size_t>(row0 + l) * groups + g0 + c]);
                    }
                }

                // ---- 5. Run the 64-wide microkernel over the L1-resident buffer ----
                for (int c = 0; c < nw; ++c) {
                    __m512i win[16];
                    for (int w = 0; w < 16; ++w) {
                        const int gwin = c * 16 + w;
                        win[w] =
                            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&win_data[gwin][0]));
                    }
                    __m512i acc = _mm512_setzero_si512();
                    for (int w = 0; w < 16; ++w) {
                        std::int32_t aw;
                        std::memcpy(&aw, act_win + 4 * (c * 16 + w), 4);
                        acc = _mm512_dpbusd_epi32(acc, win[w], _mm512_set1_epi32(aw));
                    }
                    acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * Sg[c]));
                    __m512 af = _mm512_cvtepi32_ps(acc);
                    af = _mm512_mul_ps(af, _mm512_loadu_ps(sf[c]));
                    __m512 tr = _mm512_loadu_ps(tmp16);
                    tr = _mm512_add_ps(tr, af);
                    _mm512_storeu_ps(tmp16, tr);
                }
            }

            // ---- Apply ascale and write output ----
            for (int l = 0; l < rows_this; ++l) {
                out[static_cast<std::size_t>(row0 + l) * T + t] = asc * tmp16[l];
            }
        }
    }
}

// Quantize an FP32 vector to int8 with the given per-token scale convention:
// amax -> scale = amax/127, code = clamp(round(v/scale), -127, 127); *out_scale = scale.
void quantize_i8(const float* v, int n, float* out_scale, std::int8_t* code) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) { amax = std::max(amax, std::fabs(v[i])); }
    const float sc = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    *out_scale = sc;
    for (int i = 0; i < n; ++i) {
        int c = static_cast<int>(std::lround(v[i] / sc));
        code[i] = static_cast<std::int8_t>(std::min(std::max(c, -127), 127));
    }
}

}  // namespace

namespace ninfer::ops {

void sparse_moe_512x10_cpu(const std::int16_t* x, std::int32_t T,
                           const SparseMoe512x10Geometry& geo, const SparseMoe512x10CpuWeights& w,
                           std::int16_t* destination, int thread_count) {
    const int H = geo.hidden, I = geo.inter, E = geo.n_experts, K = geo.top_k;
    if (T <= 0 || H <= 0 || I <= 0 || E <= 0 || K <= 0 || K > 16) {
        throw std::invalid_argument("sparse_moe_512x10_cpu: bad geometry");
    }
    if ((H % 64) != 0 || (I % 64) != 0) {
        throw std::invalid_argument("sparse_moe_512x10_cpu: hidden/inter must be multiples of 64");
    }

    // ---- router GEMV (FP32, exact BF16 activation) -> scores[(E+1)][T] -------------
    std::vector<float> scores(static_cast<std::size_t>(E + 1) * T);
    for (int e = 0; e <= E; ++e) {
        const float* wrow = w.router + static_cast<std::size_t>(e) * H;
        for (int t = 0; t < T; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < H; ++k) {
                acc += wrow[k] * bf16_to_f32(x[static_cast<std::size_t>(k) * T + t]);
            }
            scores[static_cast<std::size_t>(e) * T + t] = acc;
        }
    }

    // ---- top-K selection + softmax + norm_w (per token) ---------------------------
    const float kNormFloor = 6.103515625e-5f;  // 2^-14
    std::vector<int> topk_ids(static_cast<std::size_t>(T) * K);
    std::vector<float> topk_w(static_cast<std::size_t>(T) * K);
    std::vector<float> shared_logit(T);
    std::vector<float> prob(E);
    for (int t = 0; t < T; ++t) {
        float mx = -1e30f;
        for (int e = 0; e < E; ++e) { mx = std::max(mx, scores[static_cast<std::size_t>(e) * T + t]); }
        float sum = 0.0f;
        for (int e = 0; e < E; ++e) { sum += std::exp(scores[static_cast<std::size_t>(e) * T + t] - mx); }
        for (int e = 0; e < E; ++e) { prob[e] = std::exp(scores[static_cast<std::size_t>(e) * T + t] - mx) / sum; }
        std::vector<int> order;
        order.reserve(K);
        for (int rank = 0; rank < K; ++rank) {
            int best  = -1;
            float bestp = -1.0f;
            for (int e = 0; e < E; ++e) {
                if (best < 0 || prob[e] > bestp || (prob[e] == bestp && e < best)) {
                    best = e;
                    bestp = prob[e];
                }
            }
            order.push_back(best);
            prob[best] = -1.0f;
        }
        float wsum = 0.0f;
        for (int rank = 0; rank < K; ++rank) {
            wsum += std::exp(scores[static_cast<std::size_t>(order[rank]) * T + t] - mx) / sum;
        }
        const float denom = std::max(wsum, kNormFloor);
        for (int rank = 0; rank < K; ++rank) {
            const int e = order[rank];
            topk_ids[static_cast<std::size_t>(t) * K + rank] = e;
            topk_w[static_cast<std::size_t>(t) * K + rank] =
                std::exp(scores[static_cast<std::size_t>(e) * T + t] - mx) / sum / denom;
        }
        shared_logit[t] = scores[static_cast<std::size_t>(E) * T + t];
    }

    // ---- input activation quantization (once per token) ----------------------------
    std::vector<std::int8_t> aq_in(static_cast<std::size_t>(T) * H);
    std::vector<float> ascale_in(T, 1.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<float> v(H);
        for (int k = 0; k < H; ++k) { v[k] = bf16_to_f32(x[static_cast<std::size_t>(k) * T + t]); }
        quantize_i8(v.data(), H, &ascale_in[t], aq_in.data() + static_cast<std::size_t>(t) * H);
    }

    // ---- per-(t, rank) routed GEMV jobs + per-(t) shared job -----------------------
    std::vector<float> y(static_cast<std::size_t>(T) * K * H);      // [t][rank][i]
    std::vector<float> shared_y(static_cast<std::size_t>(T) * H);   // [t][i]

    // Per-expert plane strides follow the codec: Q4 packs 2 codes/byte (row stride k/2),
    // Q8 stores one i8 per code (row stride k). The FP16 scale plane is codec-independent
    // (2 B per 64-group), so its per-expert strides use the logical row counts.
    const int gu_row_bytes = (w.routed_gate_up_codec == MoeCode::Q4G64) ? (H / 2) : H;
    const int dn_row_bytes = (w.routed_down_codec == MoeCode::Q4G64) ? (I / 2) : I;
    const int sgu_row_bytes = (w.shared_gate_up_codec == MoeCode::Q4G64) ? (H / 2) : H;
    const int sdn_row_bytes = (w.shared_down_codec == MoeCode::Q4G64) ? (I / 2) : I;

    const int n_jobs = T * (K + 1);
    auto run_job = [&](int job) {
        const int t     = job / (K + 1);
        const int which = job % (K + 1);  // 0..K-1 = routed rank, K = shared
        const std::int8_t* aq_t = aq_in.data() + static_cast<std::size_t>(t) * H;
        const float* as_t       = ascale_in.data() + t;
        if (which < K) {
            const int e        = topk_ids[static_cast<std::size_t>(t) * K + which];
            const std::size_t gbase = static_cast<std::size_t>(e) * 2 * I;
            std::vector<float> gu(static_cast<std::size_t>(2 * I));
            int8_gemv(w.routed_gate_up_codec,
                      w.routed_gate_up_base + gbase * gu_row_bytes,
                      w.routed_gate_up_scales + gbase * (H / 64),
                      2 * I, H, aq_t, as_t, 1, gu.data());
            std::vector<float> act(I);
            for (int j = 0; j < I; ++j) {
                act[j] = silu_f32(gu[j]) * gu[I + j];
            }
            float asc;
            std::vector<std::int8_t> aq_act(I);
            quantize_i8(act.data(), I, &asc, aq_act.data());
            float* out_down = y.data() + static_cast<std::size_t>(t * K + which) * H;
            int8_gemv(w.routed_down_codec,
                      w.routed_down_base + static_cast<std::size_t>(e) * H * dn_row_bytes,
                      w.routed_down_scales +
                          static_cast<std::size_t>(e) * H * (I / 64),
                      H, I, aq_act.data(), &asc, 1, out_down);
        } else {
            std::vector<float> gu(static_cast<std::size_t>(2 * I));
            int8_gemv(w.shared_gate_up_codec, w.shared_gate_up_base,
                      w.shared_gate_up_scales, 2 * I, H,
                      aq_t, as_t, 1, gu.data());
            std::vector<float> sact(I);
            for (int j = 0; j < I; ++j) { sact[j] = silu_f32(gu[j]) * gu[I + j]; }
            float asc;
            std::vector<std::int8_t> saq(I);
            quantize_i8(sact.data(), I, &asc, saq.data());
            int8_gemv(w.shared_down_codec, w.shared_down_base,
                      w.shared_down_scales, H, I, saq.data(),
                      &asc, 1, shared_y.data() + static_cast<std::size_t>(t) * H);
        }
    };

    int nthreads = (thread_count > 0) ? thread_count : static_cast<int>(std::thread::hardware_concurrency());
    if (nthreads <= 0) { nthreads = 1; }
    nthreads = std::min(nthreads, n_jobs);
    if (nthreads == 1) {
        for (int job = 0; job < n_jobs; ++job) { run_job(job); }
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int h = 0; h < nthreads; ++h) {
            pool.emplace_back([&, h]() {
                for (int job = h; job < n_jobs; job += nthreads) { run_job(job); }
            });
        }
        for (auto& th : pool) { th.join(); }
    }

    // ---- merge (single-threaded, fixed rank order, shared last) --------------------
    for (int t = 0; t < T; ++t) {
        const float sgate = sigmoid_f32(shared_logit[t]);
        for (int i = 0; i < H; ++i) {
            float acc = 0.0f;
            for (int rank = 0; rank < K; ++rank) {
                acc += topk_w[static_cast<std::size_t>(t) * K + rank] *
                       y[static_cast<std::size_t>(t * K + rank) * H + i];
            }
            acc += sgate * shared_y[static_cast<std::size_t>(t) * H + i];
            const float res = bf16_to_f32(destination[static_cast<std::size_t>(i) * T + t]);
            destination[static_cast<std::size_t>(i) * T + t] = f32_to_bf16(res + acc);
        }
    }
}

}  // namespace ninfer::ops
