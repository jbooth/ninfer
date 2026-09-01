#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16LinearDecodeSchedule<Geometry>;

    const Bf16ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == 14336 && weight.k == 5120) {
        launch_geometry<Bf16GemvGeometry<14336, 5120>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 6144) {
        launch_geometry<Bf16GemvGeometry<5120, 6144>>(x, weight, out, stream);
        return;
    }
    // qwen4exp rows (T=1 tuned GEMV where the default 32-row/256-value tile fits):
    // HC down [320,10240], PLE key/value [10240,2560]/[2560,2560], output head
    // [248320,2560], indexer q/k [512,2560]/[128,2560], GDN a/b [96,2560].
    if (weight.n == 320 && weight.k == 10240) {
        launch_geometry<Bf16GemvGeometry<320, 10240>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 10240 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<10240, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 2560 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<2560, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 248320 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<248320, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 512 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<512, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 128 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<128, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 96 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<96, 2560>>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear decode: unsupported exact problem");
}

} // namespace ninfer::ops::detail
