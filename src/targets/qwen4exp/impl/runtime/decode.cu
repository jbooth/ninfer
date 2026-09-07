// qwen4exp (jbq4) decode dataflow (JM4b): the 48-layer eager, text-only forward.
//
// Layout conventions (all activations BF16 [dim, B] with dim fastest, B slowest, unless noted):
//   - ops::linear expects x [K, T] (K fastest) and produces out [N, T].
//   - QSA t-fastest: element (d, h, b) at d*H*B + h*B + b (the qsa_split_qgate / qsa_softmax
//     attention kernel layout, confirmed in qsa.cuh).
//   - rope/rmsnorm d-fastest: element (d, h, b) at d + h*head_dim + b*head_dim*H.
//   - GDN SSM: FP32 [128, 128, 48, Slots] (128x128 matrix contiguous per (h, slot)).
//   - GDN conv_states: BF16 [C, 3, Slots] (C fastest).
//   - QSA main KV: BF16, per-lane [256, 2, kv_capacity] (dim fastest, kv_heads middle, cell
//     slowest; cell j at offset j*512). QSA side cache: BF16, per-lane [128, kv_capacity].
//
// The target owns the orchestration and the layout glue (GDN qkv assembly via device copies,
// QSA head/token transposes via the kernels below, per-lane QSA attention/indexer). Numerical
// parity is deferred to JM6; the bar is compile-green + no-crash on the real artifact.

#include "runtime/decode.h"

#include "core/device.h"
#include "cpu/cpu.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hc_mix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple_mix.h"
#include "ops/kernel/ple_mix.cuh"  // PLE kernels (the ple_mix wrapper is not in the build).
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_moe_512x10_cpu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::targets::qwen4exp::detail;  // DecodeState, RoundLane, DeviceWeights, etc.

namespace {

// ---------------------------------------------------------------------------
// Geometry + layout constants.
// ---------------------------------------------------------------------------
constexpr int kLayers = 48;
constexpr int kHidden = 2560;
constexpr int kHcDim = 10240;
constexpr int kHc = 4;
constexpr int kHcLr = 320;
constexpr int kPleLayer = 1;
constexpr int kGdnQkProj = 4096;   // 16 * 128 * 2
constexpr int kGdnVzProj = 12288;  // 48 * 128 * 2
constexpr int kGdnConvCh = 10240;  // 2*2048 + 6144
constexpr int kGdnHv = 48;
constexpr int kGdnHq = 16;
constexpr int kGdnD = 128;
constexpr int kQsaQkgv = 13312;
constexpr int kQsaQ = 6144;        // 24 * 256
constexpr int kQsaK = 512;         // 2 * 256
constexpr int kQsaGate = 6144;     // 24 * 256
constexpr int kQsaV = 512;         // 2 * 256
constexpr int kQsaQHeads = 24;
constexpr int kQsaKvHeads = 2;
constexpr int kQsaHeadDim = 256;
constexpr int kIdxQuery = 512;
constexpr int kIdxKey = 128;
constexpr int kRopeDim = 64;
constexpr float kRopeTheta = 1e7f;
constexpr float kNormEps = 1e-6f;
constexpr int kGdnLayers = 36;
constexpr int kQsaLayers = 12;

[[nodiscard]] bool is_full_attn_layer(int l) noexcept { return l % 4 == 3; }
[[nodiscard]] int gdn_index(int l) noexcept { return (l / 4) * 3 + (l % 4); }
[[nodiscard]] int qsa_index(int l) noexcept { return l / 4; }

// Norm cache layout (bytes): GDN [128] * 36, then QSA [256, 256, 128, 128] * 12.
constexpr size_t kNormGdnBytes = std::size_t(kGdnLayers) * 128 * 2;
constexpr size_t kNormQsaStride = std::size_t(256 + 256 + 128 + 128) * 2;

// ---------------------------------------------------------------------------
// Target-owned layout-glue kernels (BF16).
// ---------------------------------------------------------------------------

// Transpose between the QSA t-fastest [D, H, B] (element (d,h,b) at d*H*B + h*B + b) and the
// rope/rmsnorm d-fastest [D, H, B] (element (d,h,b) at d + h*D + b*D*H). For B=1 the d-fastest
// is the head-major concat [D*H] (element (d,h,0) at d + h*D).
__global__ void transpose_t2d(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                              int D, int H, int B) {
    int b = blockIdx.z;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= D * H) return;
    int h = n % H;
    int d = n / H;
    dst[d + h * D + b * D * H] = src[d * H * B + h * B + b];
}

__global__ void transpose_d2t(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                              int D, int H, int B) {
    int b = blockIdx.z;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= D * H) return;
    int h = n % H;
    int d = n / H;
    dst[d * H * B + h * B + b] = src[d + h * D + b * D * H];
}

// Extract lane b's column from a t-fastest [D, H, B] (element (d,h,b) at d*H*B + h*B + b) into a
// contiguous t-fastest [D, H, 1] (element (d,h,0) at d*H + h).
__global__ void extract_lane_tfast(const __nv_bfloat16* __restrict__ src,
                                   __nv_bfloat16* __restrict__ dst, int D, int H, int B, int b) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= D * H) return;
    int h = n % H;
    int d = n / H;
    dst[d * H + h] = src[d * H * B + h * B + b];
}

// Place a contiguous t-fastest [D, H, 1] column into lane b of a t-fastest [D, H, B] (writing the
// d-fastest [D*H, B] at lane b: dst[d + h*D + b*D*H] = src[d*H + h]).
__global__ void place_lane_tfast(const __nv_bfloat16* __restrict__ src,
                                 __nv_bfloat16* __restrict__ dst, int D, int H, int B, int b) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= D * H) return;
    int h = n % H;
    int d = n / H;
    dst[d + h * D + b * D * H] = src[d * H + h];
}

// PLE fold-back: x[10240, B] += gated[10240, B] + conv_out[10240, B] (in place, BF16).
__global__ void foldback3(__nv_bfloat16* __restrict__ x,
                          const __nv_bfloat16* __restrict__ a,
                          const __nv_bfloat16* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]) + __bfloat162float(a[i]) + __bfloat162float(b[i]);
    x[i] = __float2bfloat16(v);
}

// Initial res_hc = repeat_4(emb): x[10240, B] = [emb; emb; emb; emb] (each [2560, B]).
__global__ void repeat4(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                        int D, int B) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= D * B) return;
    __nv_bfloat16 v = src[n];
    int b = n / D;
    int d = n % D;
    dst[d + b * 4 * D] = v;
    dst[D + d + b * 4 * D] = v;
    dst[2 * D + d + b * 4 * D] = v;
    dst[3 * D + d + b * 4 * D] = v;
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

[[nodiscard]] __nv_bfloat16* bf16p(const Tensor& t) noexcept {
    return static_cast<__nv_bfloat16*>(t.data);
}

[[nodiscard]] const __nv_bfloat16* cbf16p(const Tensor& t) noexcept {
    return static_cast<const __nv_bfloat16*>(t.data);
}

// Construct a BF16_CTRL Weight from a device BF16 Tensor (logical [N, K], dim-0 fastest).
Weight bf16_weight(const Tensor& t) {
    Weight w{};
    w.qtype = QType::BF16_CTRL;
    w.layout = QuantLayout::Contiguous;
    w.qdata = t.data;
    w.payload = t.data;
    w.payload_bytes = t.bytes();
    w.ndim = 2;
    w.n = t.ne[0];
    w.k = t.ne[1];
    w.shape[0] = t.ne[0];
    w.shape[1] = t.ne[1];
    w.padded_shape[0] = t.ne[0];
    w.padded_shape[1] = t.ne[1];
    return w;
}

void launch_transpose_t2d(cudaStream_t stream, const __nv_bfloat16* src, __nv_bfloat16* dst, int D,
                          int H, int B) {
    dim3 block(256);
    dim3 grid((D * H + 255) / 256, 1, B);
    transpose_t2d<<<grid, block, 0, stream>>>(src, dst, D, H, B);
}

void launch_transpose_d2t(cudaStream_t stream, const __nv_bfloat16* src, __nv_bfloat16* dst, int D,
                          int H, int B) {
    dim3 block(256);
    dim3 grid((D * H + 255) / 256, 1, B);
    transpose_d2t<<<grid, block, 0, stream>>>(src, dst, D, H, B);
}

void launch_repeat4(cudaStream_t stream, const __nv_bfloat16* emb, __nv_bfloat16* x, int B) {
    int n = kHidden * B;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    repeat4<<<grid, block, 0, stream>>>(emb, x, kHidden, B);
}

// PLE launch helpers (inline, bypassing the ple_mix wrapper which is not in the build).
void launch_ple_stream_gate(cudaStream_t stream, const __nv_bfloat16* key,
                            const __nv_bfloat16* query, float* out, int stream_dim, int T) {
    dim3 grid(T, kHc / stream_dim);
    ninfer::ops::ple_stream_gate_kernel<<<grid, 256, 0, stream>>>(key, query, out, stream_dim, T,
                                                                  1.0f / std::sqrt((float)stream_dim));
}

void launch_ple_gate_scale(cudaStream_t stream, const __nv_bfloat16* value, const float* gate,
                           __nv_bfloat16* out, int stream_dim, int T) {
    std::int64_t n = std::int64_t(kHc) * T;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    ninfer::ops::ple_gate_scale_kernel<<<grid, block, 0, stream>>>(value, gate, out, stream_dim, T,
                                                                   n);
}

void launch_ple_dilated_conv(cudaStream_t stream, const __nv_bfloat16* x, const __nv_bfloat16* w,
                             const __nv_bfloat16* state_in, __nv_bfloat16* state_out,
                             __nv_bfloat16* out, int C, int T) {
    std::int64_t n = std::int64_t(C) * T;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    ninfer::ops::ple_dilated_conv_kernel<<<grid, block, 0, stream>>>(x, w, state_in, out, C, T, n);
    ninfer::ops::ple_dilated_conv_state_kernel<<<grid, block, 0, stream>>>(
        state_in, x, state_out, C, T);
}

// Per-layer/per-lane tensor views over the DecodeState buffers (raw pointer arithmetic).
Tensor gdn_a_log_view(const DecodeState& s, int gdn) {
    float* p = static_cast<float*>(s.gdn_a_log.p) + gdn * 48;
    return Tensor(p, DType::FP32, {48});
}
Tensor gdn_dt_bias_view(const DecodeState& s, int gdn) {
    float* p = static_cast<float*>(s.gdn_dt_bias.p) + gdn * 48;
    return Tensor(p, DType::FP32, {48});
}
Tensor gdn_conv_w_view(const DecodeState& s, int gdn) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.gdn_conv_w.p) + std::uint64_t(gdn) * 10240 * 4;
    return Tensor(p, DType::BF16, {10240, 4});
}
Tensor gdn_conv_state_view(const DecodeState& s) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.gdn_conv_state.p);
    return Tensor(p, DType::BF16, {10240, 3, s.max_concurrency});
}
Tensor gdn_ssm_view(const DecodeState& s) {
    float* p = static_cast<float*>(s.gdn_ssm.p);
    return Tensor(p, DType::FP32, {128, 128, 48, s.max_concurrency});
}
Tensor gdn_norm_view(const DecodeState& s, int gdn) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.norm_cache.p) + gdn * 128;
    return Tensor(p, DType::BF16, {128});
}
Tensor qsa_query_norm_view(const DecodeState& s, int qsa) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.norm_cache.p) + kNormGdnBytes / 2
                       + qsa * (kNormQsaStride / 2);
    return Tensor(p, DType::BF16, {256});
}
Tensor qsa_key_norm_view(const DecodeState& s, int qsa) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.norm_cache.p) + kNormGdnBytes / 2
                       + qsa * (kNormQsaStride / 2) + 256;
    return Tensor(p, DType::BF16, {256});
}
Tensor qsa_idx_query_norm_view(const DecodeState& s, int qsa) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.norm_cache.p) + kNormGdnBytes / 2
                       + qsa * (kNormQsaStride / 2) + 512;
    return Tensor(p, DType::BF16, {128});
}
Tensor qsa_idx_key_norm_view(const DecodeState& s, int qsa) {
    __nv_bfloat16* p = static_cast<__nv_bfloat16*>(s.norm_cache.p) + kNormGdnBytes / 2
                       + qsa * (kNormQsaStride / 2) + 640;
    return Tensor(p, DType::BF16, {128});
}

// ---------------------------------------------------------------------------
// One HC low-rank mixer: xn = grouped_rmsnorm(x, norm); lo = silu(down(xn)/4);
// up_out = up(lo); mixed = mean4(xn * sigmoid(up_out)). If `inject_w` and `inject` are non-null,
// also inject = inject_w(xn). `mixed` is the block input [2560, B]; `inject` [4, B].
// ---------------------------------------------------------------------------
void hc_mix(const DeviceWeights::Hc& hc, const Tensor& x, Tensor& xn, Tensor& lo, Tensor& up_out,
            Tensor& mixed, const Tensor* inject_w, Tensor* inject, cudaStream_t stream) {
    grouped_rmsnorm(x, hc.norm, kNormEps, kHidden, xn, stream);
    linear(xn, bf16_weight(hc.down), lo, stream);
    hc_silu_div4(lo, 4.0f, lo, stream);
    linear(lo, bf16_weight(hc.up), up_out, stream);
    hc_gate_mul_mean4(xn, up_out, kHidden, mixed, stream);
    if (inject_w != nullptr && inject != nullptr) {
        linear(xn, bf16_weight(*inject_w), *inject, stream);
    }
}

// ---------------------------------------------------------------------------
// GDN (linear-attention) block: mixed [2560, B] -> block_out [2560, B].
// Per-lane state via source/dest slots = lane ids.
// ---------------------------------------------------------------------------
void gdn_block(const DeviceWeights::Layer& layer, const Tensor& mixed, Tensor& block_out,
               const DecodeState& state, int gdn, const Tensor& slots, cudaStream_t stream,
               DeviceArena& ws) {
    const auto& g = layer.gdn;
    const int B = mixed.ne[1];
    auto bf16_alloc = [&](int rows) { return ws.alloc(DType::BF16, {rows, B}); };

    Tensor qk = bf16_alloc(kGdnQkProj);
    Tensor vz = bf16_alloc(kGdnVzProj);
    Tensor qkv = bf16_alloc(kGdnConvCh);
    Tensor conv_out = bf16_alloc(kGdnConvCh);
    Tensor q_buf = bf16_alloc(kGdnD * kGdnHq);
    Tensor k_buf = bf16_alloc(kGdnD * kGdnHq);
    Tensor v_buf = bf16_alloc(kGdnD * kGdnHv);
    Tensor ab = bf16_alloc(2 * kGdnHv);
    Tensor g_buf = ws.alloc(DType::FP32, {kGdnHv, B});
    Tensor beta_buf = ws.alloc(DType::FP32, {kGdnHv, B});
    Tensor ssm_out = bf16_alloc(kGdnD * kGdnHv);
    Tensor o_normed = bf16_alloc(kGdnD * kGdnHv);

    // q|k projection -> qk [4096, B]; v|z projection -> vz [12288, B].
    linear(mixed, g.query_key, qk, stream);
    linear(mixed, g.value_z, vz, stream);

    // Assemble the conv input qkv [10240, B] = [qk (4096) | vz[0:6144] (6144)] (2 device copies).
    {
        int qk_bytes = kGdnQkProj * B * 2;
        int v_bytes = kGdnD * kGdnHv * B * 2;
        CUDA_CHECK(cudaMemcpyAsync(bf16p(qkv), bf16p(qk), qk_bytes, cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(bf16p(qkv) + kGdnQkProj * B, bf16p(vz), v_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // GDN gating: a = ab[0:48], b = ab[48:96]; g = -exp(A_log)*softplus(a+dt), beta = sigmoid(b).
    {
        Tensor ab_w = g.a_b_projection;  // BF16 [96, 2560].
        linear(mixed, bf16_weight(ab_w), ab, stream);  // [96, B].
        Tensor a = ab.slice(0, 0, kGdnHv);             // [48, B].
        Tensor b = ab.slice(0, kGdnHv, kGdnHv);        // [48, B].
        gdn_gating(a, b, gdn_a_log_view(state, gdn), gdn_dt_bias_view(state, gdn), g_buf, beta_buf,
                   stream);
    }

    // Causal width-4 conv + silu over the qkv (per-lane conv state via snapshot slots).
    {
        Tensor x3 = qkv.view({kGdnConvCh, 1, B});
        Tensor conv_out3 = conv_out.view({kGdnConvCh, 1, B});
        Tensor conv_states = gdn_conv_state_view(state);
        Tensor empty;  // valid_columns empty (all rows W=1).
        causal_conv1d_silu_snapshot(x3, gdn_conv_w_view(state, gdn), conv_states, empty, slots,
                                    slots, conv_out3, stream);
    }

    // Extract q, k, v from the conv output (3 device copies) and view as [D, H, 1, B].
    {
        int qk_bytes = kGdnD * kGdnHq * B * 2;
        int v_bytes = kGdnD * kGdnHv * B * 2;
        const __nv_bfloat16* conv_b = cbf16p(conv_out);
        CUDA_CHECK(cudaMemcpyAsync(bf16p(q_buf), conv_b, qk_bytes, cudaMemcpyDeviceToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(bf16p(k_buf), conv_b + kGdnD * kGdnHq * B, qk_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(bf16p(v_buf), conv_b + 2 * kGdnD * kGdnHq * B, v_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    Tensor q4 = q_buf.view({kGdnD, kGdnHq, 1, B});
    Tensor k4 = k_buf.view({kGdnD, kGdnHq, 1, B});
    Tensor v4 = v_buf.view({kGdnD, kGdnHv, 1, B});
    Tensor ssm = gdn_ssm_view(state);
    Tensor g4 = g_buf.view({kGdnHv, 1, B});
    Tensor beta4 = beta_buf.view({kGdnHv, 1, B});
    Tensor out4 = ssm_out.view({kGdnD, kGdnHv, 1, B});
    float scale = 1.0f / std::sqrt(128.0f);
    gated_delta_net_batch_update(q4, k4, v4, g4, beta4, scale, true, ssm, slots, slots, out4,
                                 stream);

    // Gated norm: o_normed = rmsnorm(ssm_out, gdn_norm[128], unit_offset) * sigmoid(z).
    {
        Tensor z = vz.slice(0, kGdnD * kGdnHv, kGdnD * kGdnHv);  // [6144, B]
        rmsnorm(ssm_out, gdn_norm_view(state, gdn), kNormEps, /*unit_offset=*/true, o_normed,
                stream);
        sigmoid_mul(z, o_normed, stream);
    }

    linear(o_normed, g.output, block_out, stream);
}

// ---------------------------------------------------------------------------
// QSA (full-attention) block: mixed [2560, B] -> block_out [2560, B].
// Per-lane: append k/v to the main KV, append the indexer key to the side cache,
// qsa_indexer -> topk, qsa_softmax_attention.
// ---------------------------------------------------------------------------
void qsa_block(const DeviceWeights::Layer& layer, const Tensor& mixed, Tensor& block_out,
               const DecodeState& state, int qsa, const RoundLane* lanes, int B, cudaStream_t stream,
               DeviceArena& ws) {
    const auto& q = layer.qsa;
    const int kv_capacity = state.kv_capacity;

    auto bf16_alloc = [&](int rows) { return ws.alloc(DType::BF16, {rows, B}); };

    Tensor parent = bf16_alloc(kQsaQkgv);
    Tensor q_t = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaQHeads, B});
    Tensor gate_t = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaQHeads, B});
    Tensor q_d = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaQHeads, B});
    Tensor k_d = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaKvHeads, B});
    Tensor v_d = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaKvHeads, B});
    Tensor idx_key = bf16_alloc(kIdxKey);
    Tensor idx_query = bf16_alloc(kIdxQuery);
    Tensor positions = ws.alloc(DType::I32, {B});
    Tensor out_d = bf16_alloc(kQsaQHeads * kQsaHeadDim);
    Tensor gate_d = bf16_alloc(kQsaQHeads * kQsaHeadDim);

    // Parent = linear(mixed, qkgv) -> [13312, B]. Split q|gate (t-fastest); k/v slices (d-fastest).
    linear(mixed, q.query_key_gate_value, parent, stream);
    qsa_split_qgate(parent, q_t, gate_t, stream);
    Tensor k_slice = parent.slice(0, kQsaQ, kQsaK);
    Tensor v_slice = parent.slice(0, kQsaQ + kQsaGate + kQsaK, kQsaV);
    // k/v slices are [512, B] d-fastest; view as [256, 2, B] (head-major).
    Tensor k_view = k_slice.view({kQsaHeadDim, kQsaKvHeads, B});
    Tensor v_view = v_slice.view({kQsaHeadDim, kQsaKvHeads, B});
    CUDA_CHECK(cudaMemcpyAsync(bf16p(k_d), cbf16p(k_view), k_d.bytes(), cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(bf16p(v_d), cbf16p(v_view), v_d.bytes(), cudaMemcpyDeviceToDevice,
                               stream));

    // Transpose q t-fastest -> d-fastest for norm+rope.
    launch_transpose_t2d(stream, cbf16p(q_t), bf16p(q_d), kQsaHeadDim, kQsaQHeads, B);

    // Per-head norm (d-fastest) + rope.
    rmsnorm(q_d, qsa_query_norm_view(state, qsa), kNormEps, /*unit_offset=*/false, q_d, stream);
    rmsnorm(k_d, qsa_key_norm_view(state, qsa), kNormEps, /*unit_offset=*/false, k_d, stream);
    {
        std::vector<int> host_positions(B);
        for (int b = 0; b < B; ++b) host_positions[b] = lanes[b].position;
        CUDA_CHECK(cudaMemcpyAsync(positions.data, host_positions.data(), sizeof(int) * B,
                                   cudaMemcpyHostToDevice, stream));
    }
    rope(positions, kRopeDim, kRopeTheta, q_d, k_d, stream);

    // Indexer projections (the side-cache key + query).
    linear(mixed, bf16_weight(q.indexer_key_proj), idx_key, stream);
    linear(mixed, bf16_weight(q.indexer_query_proj), idx_query, stream);

    // Transpose q d-fastest -> t-fastest for the attention.
    launch_transpose_d2t(stream, cbf16p(q_d), bf16p(q_t), kQsaHeadDim, kQsaQHeads, B);

    // Per-lane state + attention.
    __nv_bfloat16* k_cache_base = reinterpret_cast<__nv_bfloat16*>(state.qsa_kv_k.p);
    __nv_bfloat16* v_cache_base = reinterpret_cast<__nv_bfloat16*>(state.qsa_kv_v.p);
    __nv_bfloat16* side_base = reinterpret_cast<__nv_bfloat16*>(state.qsa_side.p);

    for (int b = 0; b < B; ++b) {
        const int lane = lanes[b].lane;
        const int pos = lanes[b].position;
        const int n_kv = pos + 1;
        const std::uint64_t lane_off =
            std::uint64_t(qsa * state.max_concurrency + lane) * kv_capacity;
        __nv_bfloat16* lane_k = k_cache_base + lane_off * kQsaK;
        __nv_bfloat16* lane_v = v_cache_base + lane_off * kQsaV;
        __nv_bfloat16* lane_side = side_base + lane_off * kIdxKey;

        // Append the indexer key cell (the lane's [128] from idx_key) to the side cache.
        CUDA_CHECK(cudaMemcpyAsync(lane_side + pos * kIdxKey, cbf16p(idx_key) + b * kIdxKey,
                                   kIdxKey * 2, cudaMemcpyDeviceToDevice, stream));
        // Append the k cell (the lane's [512] from k_d) and the v cell (the lane's [512] from
        // v_d) to the main KV.
        CUDA_CHECK(cudaMemcpyAsync(lane_k + pos * kQsaK, cbf16p(k_d) + b * kQsaK, kQsaK * 2,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(lane_v + pos * kQsaV, cbf16p(v_d) + b * kQsaV, kQsaV * 2,
                                   cudaMemcpyDeviceToDevice, stream));

        // Per-lane indexer: k_cache [128, n_kv], q_raw [512, 1], positions [1].
        Tensor lane_pos = ws.alloc(DType::I32, {1});
        Tensor topk = ws.alloc(DType::I32, {kQsaTopKWidth, 1});
        Tensor score = ws.alloc(DType::FP32, {(n_kv + 3) / 4, 1});
        CUDA_CHECK(cudaMemcpyAsync(lane_pos.data, &pos, sizeof(int), cudaMemcpyHostToDevice,
                                   stream));
        Tensor lane_k_cache(const_cast<void*>(static_cast<const void*>(lane_side)), DType::BF16,
                           {kIdxKey, n_kv});
        Tensor lane_q_raw(bf16p(idx_query) + b * kIdxQuery, DType::BF16, {kIdxQuery, 1});
        WorkspaceArena idx_ws(ws.alloc_bytes(qsa_indexer_workspace_capacity_bytes(n_kv, 1)));
        qsa_indexer(lane_k_cache, lane_q_raw, lane_pos, qsa_idx_key_norm_view(state, qsa),
                    qsa_idx_query_norm_view(state, qsa), topk, score, idx_ws, stream);

        // Per-lane attention: extract the lane's q to a contiguous [256, 24, 1], run, place back.
        Tensor q_lane = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaQHeads, 1});
        Tensor out_lane = ws.alloc(DType::BF16, {kQsaHeadDim, kQsaQHeads, 1});
        {
            dim3 block(256);
            dim3 grid((kQsaHeadDim * kQsaQHeads + 255) / 256);
            extract_lane_tfast<<<grid, block, 0, stream>>>(cbf16p(q_t), bf16p(q_lane),
                                                           kQsaHeadDim, kQsaQHeads, B, b);
        }
        Tensor lane_main_k(lane_k, DType::BF16, {kQsaHeadDim, kQsaKvHeads, n_kv});
        Tensor lane_main_v(lane_v, DType::BF16, {kQsaHeadDim, kQsaKvHeads, n_kv});
        AttentionHeadGeometry geometry{kQsaHeadDim, kQsaQHeads, kQsaKvHeads};
        float scale = 1.0f / std::sqrt(256.0f);
        WorkspaceArena attn_ws(ws.alloc_bytes(
            qsa_softmax_attention_workspace_capacity_bytes(geometry, 1, 1)));
        qsa_softmax_attention(q_lane, topk, lane_pos, lane_main_k, lane_main_v, n_kv, geometry,
                              scale, attn_ws, out_lane, stream);
        {
            dim3 block(256);
            dim3 grid((kQsaHeadDim * kQsaQHeads + 255) / 256);
            // out_lane is contiguous t-fastest [256, 24, 1]; place into out_d (d-fastest [6144, B])
            // at lane b.
            place_lane_tfast<<<grid, block, 0, stream>>>(cbf16p(out_lane), bf16p(out_d),
                                                         kQsaHeadDim, kQsaQHeads, B, b);
            // gate_t is t-fastest [256, 24, B]; extract lane b's column to gate_d (d-fastest
            // [6144, B]) at lane b.
            extract_lane_tfast<<<grid, block, 0, stream>>>(cbf16p(gate_t), bf16p(gate_d),
                                                           kQsaHeadDim, kQsaQHeads, B, b);
        }
    }

    // The per-lane out_d/gate_d columns are each [6144] d-fastest at lane b (contiguous). Reassemble
    // as [6144, B] d-fastest for the sigmoid_mul + output projection.
    Tensor out_d2 = out_d.view({kQsaQHeads * kQsaHeadDim, B});
    Tensor gate_d2 = gate_d.view({kQsaQHeads * kQsaHeadDim, B});
    sigmoid_mul(gate_d2, out_d2, stream);
    linear(out_d2, q.output, block_out, stream);
}

// ---------------------------------------------------------------------------
// MoE (CPU) block: mixed [2560, B] -> ffn_out [2560, B] via the page-cache MoE banks.
// ---------------------------------------------------------------------------
void moe_block(const HostArtifactView& host, int layer, const Tensor& mixed, Tensor& ffn_out,
               cudaStream_t stream, DeviceArena& ws) {
    const int B = mixed.ne[1];
    const size_t mixed_bytes = mixed.bytes();
    const size_t ffn_bytes = ffn_out.bytes();
    DeviceSpan mixed_span = ws.alloc_bytes(mixed_bytes);
    DeviceSpan ffn_span = ws.alloc_bytes(ffn_bytes);
    CUDA_CHECK(cudaMemcpyAsync(mixed_span.data, mixed.data, mixed_bytes, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(ffn_span.data, 0, ffn_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // the CPU MoE reads the D2H'd activation.

    auto map = [&host](const BankPlane& plane) -> const void* { return host.map_ptr(plane.offset); };
    SparseMoe512x10CpuWeights w{};
    w.router = static_cast<const float*>(map(host.moe_side[layer].router));
    const MoEBank& routed = host.routed[layer];
    w.routed_gate_up_codec = routed.code;
    w.routed_gate_up_base = static_cast<const uint8_t*>(map(routed.gate_up_base));
    w.routed_gate_up_scales = static_cast<const uint16_t*>(map(routed.gate_up_scales));
    w.routed_down_codec = routed.code;
    w.routed_down_base = static_cast<const uint8_t*>(map(routed.down_base));
    w.routed_down_scales = static_cast<const uint16_t*>(map(routed.down_scales));
    const MoEBank& shared = host.moe_side[layer].shared;
    w.shared_gate_up_codec = shared.code;
    w.shared_gate_up_base = static_cast<const uint8_t*>(map(shared.gate_up_base));
    w.shared_gate_up_scales = static_cast<const uint16_t*>(map(shared.gate_up_scales));
    w.shared_down_codec = shared.code;
    w.shared_down_base = static_cast<const uint8_t*>(map(shared.down_base));
    w.shared_down_scales = static_cast<const uint16_t*>(map(shared.down_scales));

    SparseMoe512x10Geometry geo{};  // defaults: 512 experts, top-10, hidden=2560, inter=640.
    const int16_t* x = static_cast<const int16_t*>(mixed_span.data);
    int16_t* dst = static_cast<int16_t*>(ffn_span.data);
    sparse_moe_512x10_cpu(x, B, geo, w, dst);

    CUDA_CHECK(cudaMemcpyAsync(ffn_out.data, ffn_span.data, ffn_bytes, cudaMemcpyHostToDevice,
                               stream));
}

// ---------------------------------------------------------------------------
// PLE block (layer 1 entry): fold-back res_hc += gated + conv_out.
// ---------------------------------------------------------------------------
void ple_block(const DeviceWeights& weights, const Tensor& x, const Tensor& ple_emb,
               const DecodeState& state, const Tensor& slots, cudaStream_t stream, DeviceArena& ws) {
    const int B = x.ne[1];
    auto bf16_alloc = [&](int rows) { return ws.alloc(DType::BF16, {rows, B}); };

    Tensor key = bf16_alloc(kHcDim);
    Tensor value = bf16_alloc(kHidden);
    Tensor key_normed = bf16_alloc(kHcDim);
    Tensor query_normed = bf16_alloc(kHcDim);
    Tensor gate = ws.alloc(DType::FP32, {kHc, B});
    Tensor gated = bf16_alloc(kHcDim);
    Tensor normalized = bf16_alloc(kHcDim);
    Tensor conv_out = bf16_alloc(kHcDim);

    // key = ple_key(emb), value = ple_value(emb).
    linear(ple_emb, bf16_weight(weights.ple_key), key, stream);
    linear(ple_emb, bf16_weight(weights.ple_value), value, stream);
    // key = grouped_rmsnorm(key, ple_norm_key); query = grouped_rmsnorm(res_hc, ple_norm_query).
    grouped_rmsnorm(key, weights.ple_norm_key, kNormEps, kHidden, key_normed, stream);
    grouped_rmsnorm(x, weights.ple_norm_query, kNormEps, kHidden, query_normed, stream);
    // gate = ple_stream_gate(key, query); gated = ple_gate_scale(value, gate).
    launch_ple_stream_gate(stream, cbf16p(key_normed), cbf16p(query_normed),
                           static_cast<float*>(gate.data), kHidden, B);
    launch_ple_gate_scale(stream, cbf16p(value), static_cast<const float*>(gate.data),
                          bf16p(gated), kHidden, B);
    // normalized = grouped_rmsnorm(gated, ple_norm_conv).
    grouped_rmsnorm(gated, weights.ple_norm_conv, kNormEps, kHidden, normalized, stream);
    // conv_out = dilated_causal_conv1d_silu(normalized, ple_conv, state) — per lane.
    {
        __nv_bfloat16* hist_base = reinterpret_cast<__nv_bfloat16*>(state.ple_conv_hist.p);
        Tensor conv_w = weights.ple_conv;  // BF16 [4, 10240] (tap-major).
        const int* host_slots = nullptr;  // read the slots from the device (D2H once).
        std::vector<int> slots_host(B);
        CUDA_CHECK(cudaMemcpyAsync(slots_host.data(), slots.data, sizeof(int) * B,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int b = 0; b < B; ++b) {
            const int lane = slots_host[b];
            Tensor x1 = normalized.slice(1, b, 1);  // [10240, 1]
            Tensor out1 = conv_out.slice(1, b, 1);  // [10240, 1]
            __nv_bfloat16* hist = hist_base + std::uint64_t(lane) * kHcDim * 9;
            launch_ple_dilated_conv(stream, cbf16p(x1), cbf16p(conv_w), hist, hist,
                                    bf16p(out1), kHcDim, 1);
        }
        (void)host_slots;
    }
    // Fold-back: x += gated + conv_out.
    int n = kHcDim * B;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    foldback3<<<grid, block, 0, stream>>>(bf16p(x), cbf16p(gated), cbf16p(conv_out), n);
}

}  // namespace

namespace ninfer::targets::qwen4exp::detail {

// ---------------------------------------------------------------------------
// Load-time transforms: GDN A_log inversion, conv transpose+cast, FP32->BF16 norm caches.
// ---------------------------------------------------------------------------
void apply_load_time_transforms(DecodeState& state, const ModelView& model, cudaStream_t stream) {
    const auto& w = model.weights;
    std::vector<float> host(10240 * 4);
    std::vector<__nv_bfloat16> host_bf16(10240 * 4);

    // GDN A_log inversion (36 layers): A_raw = log(-a_log_stored).
    {
        std::vector<float> a(48);
        std::vector<float> a_raw(48);
        for (int l = 0; l < kLayers; ++l) {
            if (is_full_attn_layer(l)) continue;
            int gdn = gdn_index(l);
            const float* src = reinterpret_cast<const float*>(w.per_layer[l].gdn.a_log.data);
            CUDA_CHECK(cudaMemcpyAsync(a.data(), src, sizeof(float) * 48, cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < 48; ++i) a_raw[i] = std::log(-a[i]);
            float* dst = reinterpret_cast<float*>(state.gdn_a_log.p) + std::uint64_t(gdn) * 48;
            CUDA_CHECK(cudaMemcpyAsync(dst, a_raw.data(), sizeof(float) * 48,
                                       cudaMemcpyHostToDevice, stream));
        }
    }

    // GDN dt_bias copy (36 layers): raw FP32 [48].
    {
        for (int l = 0; l < kLayers; ++l) {
            if (is_full_attn_layer(l)) continue;
            int gdn = gdn_index(l);
            const float* src = reinterpret_cast<const float*>(w.per_layer[l].gdn.dt_bias.data);
            float* dst = reinterpret_cast<float*>(state.gdn_dt_bias.p) + std::uint64_t(gdn) * 48;
            CUDA_CHECK(cudaMemcpyAsync(dst, src, sizeof(float) * 48, cudaMemcpyDeviceToDevice,
                                       stream));
        }
    }

    // GDN conv transpose+cast (36 layers): [4, 10240] FP32 -> [10240, 4] BF16.
    {
        for (int l = 0; l < kLayers; ++l) {
            if (is_full_attn_layer(l)) continue;
            int gdn = gdn_index(l);
            const float* src =
                reinterpret_cast<const float*>(w.per_layer[l].gdn.convolution.data);
            CUDA_CHECK(cudaMemcpyAsync(host.data(), src, sizeof(float) * 4 * 10240,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int c = 0; c < 10240; ++c)
                for (int t = 0; t < 4; ++t)
                    host_bf16[c * 4 + t] = __float2bfloat16(host[t * 10240 + c]);
            __nv_bfloat16* dst =
                reinterpret_cast<__nv_bfloat16*>(state.gdn_conv_w.p) +
                std::uint64_t(gdn) * 10240 * 4;
            CUDA_CHECK(cudaMemcpyAsync(dst, host_bf16.data(), sizeof(__nv_bfloat16) * 10240 * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
    }

    // FP32 -> BF16 norm caches: GDN [128] (36 layers), QSA query/key [256] + indexer [128]
    // (12 layers).
    auto cast_fp32_to_bf16 = [&](const Tensor& src, void* dst, int n) {
        std::vector<float> f(n);
        CUDA_CHECK(cudaMemcpyAsync(f.data(), src.data, sizeof(float) * n, cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<__nv_bfloat16> bf(n);
        for (int i = 0; i < n; ++i) bf[i] = __float2bfloat16(f[i]);
        CUDA_CHECK(cudaMemcpyAsync(dst, bf.data(), sizeof(__nv_bfloat16) * n,
                                   cudaMemcpyHostToDevice, stream));
    };
    {
        char* base = static_cast<char*>(state.norm_cache.p);
        for (int l = 0; l < kLayers; ++l) {
            if (is_full_attn_layer(l)) {
                int qsa = qsa_index(l);
                char* q = base + kNormGdnBytes + qsa * kNormQsaStride;
                cast_fp32_to_bf16(w.per_layer[l].qsa.query_norm, q, 256);
                cast_fp32_to_bf16(w.per_layer[l].qsa.key_norm, q + 256 * 2, 256);
                cast_fp32_to_bf16(w.per_layer[l].qsa.indexer_query_norm, q + 512 * 2, 128);
                cast_fp32_to_bf16(w.per_layer[l].qsa.indexer_key_norm, q + 640 * 2, 128);
            } else {
                int gdn = gdn_index(l);
                char* q = base + gdn * 128 * 2;
                cast_fp32_to_bf16(w.per_layer[l].gdn.norm, q, 128);
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<TokenId> run_decode_round(DecodeState& state, const ModelView& model,
                                      const HostArtifactView& host, cudaStream_t stream,
                                      const std::span<const RoundLane>& round) {
    const int B = static_cast<int>(round.size());
    const auto& w = model.weights;

    DeviceArena ws(DeviceSpan{state.scratch.p, state.scratch.bytes});
    ws.reset();

    // Persistent round buffers.
    Tensor x = ws.alloc(DType::BF16, {kHcDim, B});
    Tensor x_new = ws.alloc(DType::BF16, {kHcDim, B});
    Tensor emb = ws.alloc(DType::BF16, {kHidden, B});
    Tensor logits = ws.alloc(DType::BF16, {vocab, B});

    // Per-lane slots = lane ids; positions = the lane's processed-token position.
    std::vector<int> host_slots(B);
    std::vector<int> host_positions(B);
    for (int b = 0; b < B; ++b) {
        host_slots[b] = round[b].lane;
        host_positions[b] = round[b].position;
    }
    Tensor slots = ws.alloc(DType::I32, {B});
    CUDA_CHECK(cudaMemcpyAsync(slots.data, host_slots.data(), sizeof(int) * B,
                               cudaMemcpyHostToDevice, stream));

    // Ingress: token embedding gather [hidden, B], repeat_4 -> res_hc.
    {
        std::vector<TokenId> ids(B);
        for (int b = 0; b < B; ++b) ids[b] = round[b].sequence.back();
        gather_token_embedding(host, ids.data(), B, static_cast<std::byte*>(emb.data), stream);
        launch_repeat4(stream, cbf16p(emb), bf16p(x), B);
    }

    // PLE embedding (layer 1): gather [hidden, B].
    Tensor ple_emb = ws.alloc(DType::BF16, {kHidden, B});
    {
        std::vector<const TokenId*> seqs(B);
        for (int b = 0; b < B; ++b) seqs[b] = round[b].sequence.data();
        gather_ple_layer1_batch(host, seqs.data(), host_positions.data(), B,
                                static_cast<std::byte*>(ple_emb.data), stream);
    }

    // The 48-layer loop.
    for (int l = 0; l < kLayers; ++l) {
        auto layer_scope = ws.scope();
        auto& layer = w.per_layer[l];

        // PLE (layer 1 entry): fold-back.
        if (l == kPleLayer) {
            ple_block(w, x, ple_emb, state, slots, stream, ws);
        }

        // HC attn mixer.
        Tensor xn = ws.alloc(DType::BF16, {kHcDim, B});
        Tensor lo = ws.alloc(DType::BF16, {kHcLr, B});
        Tensor up_out = ws.alloc(DType::BF16, {kHcDim, B});
        Tensor mixed = ws.alloc(DType::BF16, {kHidden, B});
        Tensor inject = ws.alloc(DType::BF16, {kHc, B});
        hc_mix(layer.attn_hc, x, xn, lo, up_out, mixed, &layer.attn_hc.inject, &inject, stream);

        // GDN/QSA block.
        Tensor block_out = ws.alloc(DType::BF16, {kHidden, B});
        if (is_full_attn_layer(l)) {
            qsa_block(layer, mixed, block_out, state, qsa_index(l), round.data(), B, stream, ws);
        } else {
            gdn_block(layer, mixed, block_out, state, gdn_index(l), slots, stream, ws);
        }

        // HC combine (attn).
        hc_combine(x, block_out, inject, kHidden, x_new, stream);
        std::swap(x, x_new);

        // HC ffn mixer.
        Tensor xn2 = ws.alloc(DType::BF16, {kHcDim, B});
        Tensor lo2 = ws.alloc(DType::BF16, {kHcLr, B});
        Tensor up_out2 = ws.alloc(DType::BF16, {kHcDim, B});
        Tensor mixed2 = ws.alloc(DType::BF16, {kHidden, B});
        Tensor inject2 = ws.alloc(DType::BF16, {kHc, B});
        hc_mix(layer.ffn_hc, x, xn2, lo2, up_out2, mixed2, &layer.ffn_hc.inject, &inject2, stream);

        // MoE (CPU) block.
        Tensor ffn_out = ws.alloc(DType::BF16, {kHidden, B});
        moe_block(host, l, mixed2, ffn_out, stream, ws);

        // HC combine (ffn).
        hc_combine(x, ffn_out, inject2, kHidden, x_new, stream);
        std::swap(x, x_new);
    }

    // Output head: output_hc mixer (no inject) -> linear output_head -> logits.
    {
        Tensor xn = ws.alloc(DType::BF16, {kHcDim, B});
        Tensor lo = ws.alloc(DType::BF16, {kHcLr, B});
        Tensor up_out = ws.alloc(DType::BF16, {kHcDim, B});
        Tensor mixed = ws.alloc(DType::BF16, {kHidden, B});
        hc_mix(w.output_hc, x, xn, lo, up_out, mixed, /*inject_w=*/nullptr, /*inject=*/nullptr,
               stream);
        linear(mixed, bf16_weight(w.output_head), logits, stream);
    }

    // Sampling (greedy for the no-crash bar).
    std::vector<TokenId> out(B, 0);
    {
        Tensor out_t = ws.alloc(DType::I32, {B});
        Tensor logical = ws.alloc(DType::I32, {B});
        CUDA_CHECK(cudaMemcpyAsync(logical.data, host_positions.data(), sizeof(int) * B,
                                   cudaMemcpyHostToDevice, stream));
        std::vector<SamplingConfig> configs(B);  // greedy (temperature 0).
        DeviceSpan configs_span = ws.alloc_bytes(sizeof(SamplingConfig) * B);
        CUDA_CHECK(cudaMemcpyAsync(configs_span.data, configs.data(), sizeof(SamplingConfig) * B,
                                   cudaMemcpyHostToDevice, stream));
        size_t cap = sampling_workspace_capacity_bytes(vocab, 1, B);
        WorkspaceArena sample_ws(ws.alloc_bytes(cap));
        sample(logits, out_t, vocab, static_cast<const SamplingConfig*>(configs_span.data),
               logical, kSamplePurposeDecode, sample_ws, stream);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), out_t.data, sizeof(TokenId) * B,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return out;
}

}  // namespace ninfer::targets::qwen4exp::detail
