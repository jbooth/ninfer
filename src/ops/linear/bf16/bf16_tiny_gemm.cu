#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_tiny_gemm.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

template <int NRows, int kThreads>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    bf16_tiny_gemm_kernel<NRows, kThreads>
        <<<static_cast<unsigned>(t), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data),
            weight.k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_bf16_tiny(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // v1 routes only the qwen4exp HC inject row [4, 10240], which fits no GEMV/MMA tile family
    // (M=4 is below the smallest MMA block row, and the GEMV row groups are 32-wide).
    if (weight.n == 4 && weight.k == 10240) {
        launch_geometry<4, 128>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear tiny-n: unsupported exact problem");
}

} // namespace ninfer::ops::detail
