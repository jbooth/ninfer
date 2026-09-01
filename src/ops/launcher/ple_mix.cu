// ninfer::ops - qwen4exp PLE kernel launchers.
#include "ops/launcher/ple_mix.h"

#include "ops/common/math.h"
#include "ops/kernel/ple_mix.cuh"
#include "core/device.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

constexpr std::int32_t kGridStrideBlocks = 1024;

std::uint32_t grid_for(std::int64_t n) {
    return static_cast<std::uint32_t>(
        std::min<std::int64_t>(div_up(n, std::int64_t{256}), kGridStrideBlocks));
}
} // namespace

void ple_stream_gate_launch(const Tensor& key, const Tensor& query, Tensor& out,
                            cudaStream_t stream) {
    const std::int32_t d      = key.ne[0];
    const std::int32_t tokens = key.ne[1];
    const std::int32_t hc     = out.ne[0];
    const std::int32_t stream_dim = d / hc;
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(hc), 1u);
    ple_stream_gate_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(key.data),
        static_cast<const __nv_bfloat16*>(query.data), static_cast<float*>(out.data), stream_dim,
        tokens, 1.0F / std::sqrt(static_cast<float>(stream_dim)));
    CUDA_CHECK(cudaGetLastError());
}

void ple_gate_scale_launch(const Tensor& value, const Tensor& gate, Tensor& out, cudaStream_t stream) {
    const std::int32_t stream_dim = value.ne[0];
    const std::int32_t tokens     = value.ne[1];
    const std::int64_t n          = out.numel();
    ple_gate_scale_kernel<<<grid_for(n), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(value.data), static_cast<const float*>(gate.data),
        static_cast<__nv_bfloat16*>(out.data), stream_dim, tokens, n);
    CUDA_CHECK(cudaGetLastError());
}

void dilated_causal_conv1d_silu_launch(const Tensor& x, const Tensor& weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& out, cudaStream_t stream) {
    const std::int32_t C      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int64_t n      = x.numel();
    ple_dilated_conv_kernel<<<grid_for(n), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<__nv_bfloat16*>(out.data), C, tokens, n);
    CUDA_CHECK(cudaGetLastError());
    ple_dilated_conv_state_kernel<<<grid_for(C), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(conv_state_in.data),
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<__nv_bfloat16*>(conv_state_out.data), C, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
