#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("bf16 linear: unsupported shape or T"); }

    // The tuned exact-problem routes (control/output projections) keep their measured schedules.
    if ((n == 14336 && k == 5120) || (n == 5120 && k == 6144)) {
        if (t == 1) { return launch_bf16_decode; }
        const std::int32_t small_t_end =
            n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
        if (t <= small_t_end) { return launch_bf16_small_t; }
        return launch_bf16_mma;
    }

    // qwen4exp (jbq4) dense projections. The HC inject row [4, 10240] fits no GEMV/MMA tile family
    // and uses the tiny-n route; every other jbq4 dense shape routes to the shape-generic SIMT
    // GEMM (runtime n/k/t). Both are decode shapes (T = step lane count, <= 8).
    if (n == 4 && k == 10240) { return launch_bf16_tiny; }
    const bool jbq4_shape =
        (n == 10240 && k == 320) || (n == 320 && k == 10240) || (n == 96 && k == 2560) ||
        (n == 512 && k == 2560) || (n == 128 && k == 2560) || (n == 248320 && k == 2560) ||
        (n == 10240 && k == 2560) || (n == 2560 && k == 2560);
    if (jbq4_shape) { return launch_bf16_generic; }

    throw std::invalid_argument("bf16 linear: unsupported shape or T");
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
