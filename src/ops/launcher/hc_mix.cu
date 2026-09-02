// ninfer::ops - hyper-connection (qwen4exp) kernel launchers.
#include "ops/launcher/hc_mix.h"

#include "ops/common/math.h"
#include "ops/kernel/hc_mix.cuh"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

constexpr std::int32_t kBlocks = 1024;

std::uint32_t grid_for(std::int64_t n) {
    return static_cast<std::uint32_t>(
        std::min<std::int64_t>(div_up(n, std::int64_t{256}), kBlocks));
}
} // namespace

void grouped_rmsnorm_launch(const Tensor& x, const Tensor& gamma, float eps, std::int32_t stream_dim,
                            Tensor& out, cudaStream_t stream) {
    const std::int32_t d      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t hc     = d / stream_dim;
    if (hc <= 0) {
        throw std::invalid_argument("grouped_rmsnorm: stream_dim must divide D");
    }
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(hc), 1u);
    grouped_rmsnorm_kernel<<<grid, 1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(gamma.data),
        static_cast<__nv_bfloat16*>(out.data), stream_dim, tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

void hc_silu_div4_launch(const Tensor& x, float div, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = x.numel();
    hc_silu_div4_kernel<<<grid_for(n), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), div, n);
    CUDA_CHECK(cudaGetLastError());
}

void hc_gate_mul_mean4_launch(const Tensor& xn, const Tensor& gate, std::int32_t stream_dim,
                              Tensor& out, cudaStream_t stream) {
    const std::int32_t d      = xn.ne[0];
    const std::int32_t tokens = xn.ne[1];
    const std::int32_t hc     = d / stream_dim;
    if (hc <= 0) {
        throw std::invalid_argument("hc_gate_mul_mean4: stream_dim must divide D");
    }
    const dim3 grid(static_cast<unsigned>(stream_dim), static_cast<unsigned>(tokens), 1u);
    hc_gate_mul_mean4_kernel<<<grid, 1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(xn.data), static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(out.data), stream_dim, tokens, hc);
    CUDA_CHECK(cudaGetLastError());
}

void hc_combine_launch(const Tensor& residual, const Tensor& block, const Tensor& inject,
                       std::int32_t stream_dim, Tensor& out, cudaStream_t stream) {
    const std::int32_t d      = residual.ne[0];
    const std::int32_t tokens = residual.ne[1];
    const std::int32_t hc     = inject.ne[0];
    if (hc <= 0 || d % stream_dim != 0) {
        throw std::invalid_argument("hc_combine: stream_dim must divide D");
    }
    const std::int64_t n = d * static_cast<std::int64_t>(tokens);
    hc_combine_kernel<<<grid_for(n), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data),
        static_cast<const __nv_bfloat16*>(block.data),
        static_cast<const __nv_bfloat16*>(inject.data), static_cast<__nv_bfloat16*>(out.data),
        stream_dim, hc, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
