#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported T");
    }
    // Existing production rows keep the measured small-T route.
    if ((n == 14336 && k == 5120) || (n == 5120 && k == 6144)) {
        if (t == 1) { return launch_bf16_decode; }
        const std::int32_t small_t_end =
            n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
        if (t <= small_t_end) { return launch_bf16_small_t; }
        return launch_bf16_mma;
    }
    // qwen4exp rows. Correctness-first routing: the tuned T=1 GEMV where its 32-row/256-value
    // tiles fit, the MMA GEMM otherwise, and the tiny-n SIMT kernel for the HC inject row
    // [4,10240] that fits no tile family. Per-geometry small-T schedule tuning is a later pass.
    switch (k) {
    case 2560:
        switch (n) {
        case 10240:  // PLE key
        case 2560:   // PLE value
        case 248320: // output head
        case 512:    // full-attention indexer q
        case 128:    // full-attention indexer k
        case 96:     // GDN a/b projection
            if (t == 1) { return launch_bf16_decode; }
            return launch_bf16_mma;
        default:
            break;
        }
        break;
    case 10240:
        if (n == 320) { // HC down
            if (t == 1) { return launch_bf16_decode; }
            return launch_bf16_mma;
        }
        if (n == 4) { return launch_bf16_tiny; } // HC inject
        break;
    case 320:
        if (n == 10240) { return launch_bf16_mma; } // HC up (k=320 fits no GEMV phase tiling)
        break;
    default:
        break;
    }
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
