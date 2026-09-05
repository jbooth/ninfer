#include "ops/linear/bf16/bf16_generic.cuh"

#include "core/device.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// The generic route is used for decode (T = the step's lane count, <= 8); keep a margin of 16 so
// a wider decode batch still lands on the same shape-generic kernel.
inline constexpr std::int32_t kBf16GenericMaxTokens = 16;
inline constexpr int kBf16GenericWarps = 8;

void launch_bf16_generic(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = weight.n;
    const std::int32_t k = weight.k;
    const std::int32_t t = x.ne[1];
    if (n <= 0 || k <= 0 || t <= 0 || t > kBf16GenericMaxTokens) {
        throw std::invalid_argument("bf16 linear generic: unsupported shape or T");
    }
    const dim3 grid(static_cast<unsigned>((n + kBf16GenericWarps - 1) / kBf16GenericWarps));
    bf16_generic_gemm_kernel<kBf16GenericMaxTokens, kBf16GenericWarps>
        <<<grid, kBf16GenericWarps * kWarpSize, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), n,
            k, t);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
