#pragma once

// ninfer::ops::detail - private launch prototypes for the qwen4exp PLE kernels.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void ple_stream_gate_launch(const Tensor& key, const Tensor& query, Tensor& out, cudaStream_t stream);
void ple_gate_scale_launch(const Tensor& value, const Tensor& gate, Tensor& out, cudaStream_t stream);
void dilated_causal_conv1d_silu_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
