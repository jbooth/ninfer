#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * PLE per-stream softsign gate (qwen4exp personalization-layer composite, first half). For each
 * stream s in [0, D / stream_dim) and token t:
 *
 *   dot_{s,t} = (1/sqrt(stream_dim)) * sum_d key[s*stream_dim + d, t] * query[s*stream_dim + d, t]
 *   out[s, t] = sigmoid(sgn(dot) * sqrt(clamp(|dot|, 1e-6, +inf)))
 *
 * `key` and `query` are same-shaped contiguous BF16 [D, T] with T >= 1 and D = hc * stream_dim;
 * `out` is contiguous FP32 [hc, T] (hc = D / stream_dim). All three tensors are non-overlapping.
 * The oracle evaluates the complete formula naively in FP64 from the represented BF16 inputs;
 * the FP32 output is promoted and compared directly. The dot accumulation and reduction
 * association are implementation choices (one FP32 A16 reduction profile). No workspace or
 * persistent state.
 */
void ple_stream_gate(const Tensor& key, const Tensor& query, Tensor& out, cudaStream_t stream);

/**
 * PLE gated value broadcast (qwen4exp personalization-layer composite, second half). Given the
 * value projection `value` (contiguous BF16 [S, T]) and the per-stream gate `gate` (contiguous
 * FP32 [hc, T]) with D = hc * S:
 *
 *   out[s*S + d, t] = value[d, t] * gate[s, t]
 *
 * `out` is contiguous BF16 [D, T]; all three tensors are non-overlapping. The product is
 * evaluated in the implementation's internal precision; the oracle evaluates the formula
 * naively in FP64 from the represented inputs and the BF16 output is promoted and compared
 * directly. No workspace or persistent state.
 */
void ple_gate_scale(const Tensor& value, const Tensor& gate, Tensor& out, cudaStream_t stream);

/**
 * Depthwise causal width-4 convolution dilated by 3, followed by SiLU (qwen4exp PLE history
 * tap). Let u[c,-9..-1] be the nine values in the input state (oldest to newest) and u[c,t] =
 * x[c,t] for t >= 0. Then
 *
 *   ideal[c,t] = SiLU(weight[0,C+c] * u[c,t-9] + weight[1,C+c] * u[c,t-6] +
 *                     weight[2,C+c] * u[c,t-3] + weight[3,C+c] * u[c,t])
 *
 * `x` and `out` are contiguous BF16 [C,T], `weight` is contiguous BF16 [4,C] (tap-major, the
 * stored artifact layout), and a state is contiguous BF16 [C,9] ordered oldest to newest.
 * T may be any positive value. The oracle evaluates `ideal` naively in FP64 from the
 * represented inputs; the BF16 output is promoted and compared directly with that result.
 * Output storage rounding belongs to the Op's numerical criterion, not the oracle. Kernel
 * accumulator precision is an implementation choice. Input, weight, output, and state storage
 * do not overlap except for the explicitly allowed exact alias between state input and state
 * output. No caller workspace is used.
 */

// Reads conv_state as the initial window and replaces it with the trailing width-9 window of
// concat(conv_state, x).
void dilated_causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                Tensor& out, cudaStream_t stream);

// Distinct-state form. conv_state_in and conv_state_out may be disjoint or exactly the same
// storage; conv_state_out receives the trailing width-9 window of concat(conv_state_in, x).
void dilated_causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
