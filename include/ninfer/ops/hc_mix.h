#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/**
 * Grouped RMSNorm (qwen4exp hyper-connection pre-norm). For each stream s in
 * [0, D / stream_dim) and token t:
 *
 *   inv_{s,t} = 1 / sqrt((1/stream_dim) * sum_d x[s*stream_dim + d, t]^2 + eps)
 *   out[s*stream_dim + d, t] = x[s*stream_dim + d, t] * inv_{s,t} * gamma[s*stream_dim + d]
 *
 * `x` and `out` are same-shaped contiguous BF16 [D, T] with T >= 1; `gamma` is contiguous FP32
 * [D] (the converter has folded 1 + w into the stored value); `stream_dim` divides D; `eps` is
 * positive and finite. Inputs and output must not overlap. The oracle evaluates the formula
 * naively in FP64 from the represented inputs (the exact stored FP32 gamma, not a BF16
 * approximation). The BF16 output is promoted and compared directly; reduction association and
 * staging precision are implementation choices. No workspace or persistent state.
 */
void grouped_rmsnorm(const Tensor& x, const Tensor& gamma, float eps, std::int32_t stream_dim,
                     Tensor& out, cudaStream_t stream);

/**
 * Hyper-connection low-rank gate activation: out = silu(x / div) applied elementwise, where
 * silu(v) = v / (1 + exp(-v)). `x` and `out` are same-shaped contiguous BF16 with T >= 1;
 * `div` is positive and finite. No overlap. Oracle: naive FP64 of the represented BF16 values.
 */
void hc_silu_div4(const Tensor& x, float div, Tensor& out, cudaStream_t stream);

/**
 * Hyper-connection gate-mul-stream-mean (K2). Given the grouped-normed wide stream `xn` and the
 * pre-sigmoid up projection `gate`, both contiguous BF16 [D, T]:
 *
 *   out[d, t] = (1/hc) * sum_s xn[s*stream_dim + d, t] * sigmoid(gate[s*stream_dim + d, t])
 *
 * with D = hc * stream_dim and `out` contiguous BF16 [stream_dim, T]. The sigmoid is evaluated
 * in the implementation's internal precision (it is an intermediate, not an observable output);
 * the oracle evaluates the complete formula naively in FP64 from the represented inputs.
 * No overlap. No workspace or persistent state.
 */
void hc_gate_mul_mean4(const Tensor& xn, const Tensor& gate, std::int32_t stream_dim, Tensor& out,
                       cudaStream_t stream);

/**
 * Hyper-connection residual combine (K2). Given the wide residual `residual` (BF16 [D, T]),
 * the collapsed block output `block` (BF16 [stream_dim, T]), and the raw injection projection
 * `inject` (BF16 [hc, T]) with D = hc * stream_dim:
 *
 *   w[s, t]              = 2 * sigmoid(inject[s, t] / hc)
 *   out[s*stream_dim+d,t] = residual[s*stream_dim + d, t] + block[d, t] * w[s, t]
 *
 * (2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add.)
 * `out` is contiguous BF16 [D, T]; all inputs and output are non-overlapping. The oracle
 * evaluates the formula naively in FP64 from the represented inputs. No workspace or persistent
 * state.
 */
void hc_combine(const Tensor& residual, const Tensor& block, const Tensor& inject,
                std::int32_t stream_dim, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
