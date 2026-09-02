#pragma once

// ninfer::ops::detail - private launch prototypes for the qwen4exp hyper-connection kernels.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_rmsnorm_launch(const Tensor& x, const Tensor& gamma, float eps, std::int32_t stream_dim,
                            Tensor& out, cudaStream_t stream);
void hc_silu_div4_launch(const Tensor& x, float div, Tensor& out, cudaStream_t stream);
void hc_gate_mul_mean4_launch(const Tensor& xn, const Tensor& gate, std::int32_t stream_dim,
                              Tensor& out, cudaStream_t stream);
void hc_combine_launch(const Tensor& residual, const Tensor& block, const Tensor& inject,
                       std::int32_t stream_dim, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
