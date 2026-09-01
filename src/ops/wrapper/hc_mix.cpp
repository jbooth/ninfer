// ninfer::ops - hyper-connection (qwen4exp) wrapper: public API validation and launch dispatch.
#include "ninfer/ops/hc_mix.h"

#include "ops/launcher/hc_mix.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct Shape2 {
    std::int32_t rows;
    std::int32_t tokens;
};

Shape2 require_2d(const Tensor& t, const char* label) {
    for (int d = 2; d < 4; ++d) {
        if (t.ne[d] != 1) {
            throw std::invalid_argument(std::string("hc_mix: ") + label +
                                        " must be 2-D [rows, tokens]");
        }
    }
    if (t.ne[0] < 1 || t.ne[1] < 1) {
        throw std::invalid_argument(std::string("hc_mix: ") + label + " rows/tokens must be >= 1");
    }
    if (t.numel() > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error(std::string("hc_mix: ") + label + " exceeds the grid domain");
    }
    if (!t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("hc_mix: ") + label + " must be contiguous");
    }
    return {static_cast<std::int32_t>(t.ne[0]), static_cast<std::int32_t>(t.ne[1])};
}

void require_same2(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("hc_mix: ") + b_label +
                                        " shape must match the primary input");
        }
    }
}

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("hc_mix: ") + label + " must be BF16");
    }
}

void require_f32(const Tensor& t, const char* label) {
    if (t.dtype != DType::FP32) {
        throw std::invalid_argument(std::string("hc_mix: ") + label + " must be FP32");
    }
}

void require_positive_finite(float v, const char* label) {
    if (!(v > 0.0f) || !std::isfinite(v)) {
        throw std::invalid_argument(std::string("hc_mix: ") + label + " must be positive and finite");
    }
}
} // namespace

void grouped_rmsnorm(const Tensor& x, const Tensor& gamma, float eps, std::int32_t stream_dim,
                     Tensor& out, cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(out, "out");
    require_f32(gamma, "gamma");
    require_positive_finite(eps, "eps");
    const Shape2 xy = require_2d(x, "x");
    if (gamma.ne[0] != x.ne[0] || gamma.ne[1] != 1 || gamma.ne[2] != 1 || gamma.ne[3] != 1 ||
        !gamma.is_contiguous() || gamma.data == nullptr) {
        throw std::invalid_argument("grouped_rmsnorm: gamma must be FP32 [D] with ne[0] == D");
    }
    require_same2(x, out, "out");
    if (stream_dim <= 0 || xy.rows % stream_dim != 0) {
        throw std::invalid_argument("grouped_rmsnorm: stream_dim must divide D");
    }
    detail::grouped_rmsnorm_launch(x, gamma, eps, stream_dim, out, stream);
}

void hc_silu_div4(const Tensor& x, float div, Tensor& out, cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(out, "out");
    require_positive_finite(div, "div");
    require_2d(x, "x");
    require_same2(x, out, "out");
    detail::hc_silu_div4_launch(x, div, out, stream);
}

void hc_gate_mul_mean4(const Tensor& xn, const Tensor& gate, std::int32_t stream_dim, Tensor& out,
                       cudaStream_t stream) {
    require_bf16(xn, "xn");
    require_bf16(gate, "gate");
    require_bf16(out, "out");
    const Shape2 xy = require_2d(xn, "xn");
    require_same2(xn, gate, "gate");
    if (stream_dim <= 0 || xy.rows % stream_dim != 0) {
        throw std::invalid_argument("hc_gate_mul_mean4: stream_dim must divide D");
    }
    if (out.ne[0] != stream_dim) {
        throw std::invalid_argument("hc_gate_mul_mean4: out rows must equal stream_dim");
    }
    if (out.ne[1] != xy.tokens || out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() ||
        out.data == nullptr) {
        throw std::invalid_argument("hc_gate_mul_mean4: out must be [stream_dim, T]");
    }
    detail::hc_gate_mul_mean4_launch(xn, gate, stream_dim, out, stream);
}

void hc_combine(const Tensor& residual, const Tensor& block, const Tensor& inject,
                std::int32_t stream_dim, Tensor& out, cudaStream_t stream) {
    require_bf16(residual, "residual");
    require_bf16(block, "block");
    require_bf16(inject, "inject");
    require_bf16(out, "out");
    const Shape2 xy = require_2d(residual, "residual");
    require_same2(residual, out, "out");
    if (stream_dim <= 0 || xy.rows % stream_dim != 0) {
        throw std::invalid_argument("hc_combine: stream_dim must divide D");
    }
    const std::int32_t hc = xy.rows / stream_dim;
    if (inject.ne[0] != hc || inject.ne[1] != xy.tokens || inject.ne[2] != 1 || inject.ne[3] != 1 ||
        !inject.is_contiguous() || inject.data == nullptr) {
        throw std::invalid_argument("hc_combine: inject must be [hc, T]");
    }
    if (block.ne[0] != stream_dim || block.ne[1] != xy.tokens || block.ne[2] != 1 ||
        block.ne[3] != 1 || !block.is_contiguous() || block.data == nullptr) {
        throw std::invalid_argument("hc_combine: block must be [stream_dim, T]");
    }
    detail::hc_combine_launch(residual, block, inject, stream_dim, out, stream);
}

} // namespace ninfer::ops
