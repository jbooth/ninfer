// ninfer::ops - qwen4exp PLE wrapper: public API validation and launch dispatch.
#include "ninfer/ops/ple_mix.h"

#include "ops/launcher/ple_mix.h"

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
            throw std::invalid_argument(std::string("ple_mix: ") + label +
                                        " must be 2-D [rows, tokens]");
        }
    }
    if (t.ne[0] < 1 || t.ne[1] < 1) {
        throw std::invalid_argument(std::string("ple_mix: ") + label + " rows/tokens must be >= 1");
    }
    if (t.numel() > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error(std::string("ple_mix: ") + label + " exceeds the grid domain");
    }
    if (!t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("ple_mix: ") + label + " must be contiguous");
    }
    return {static_cast<std::int32_t>(t.ne[0]), static_cast<std::int32_t>(t.ne[1])};
}

void require_same2(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("ple_mix: ") + b_label +
                                        " shape must match the primary input");
        }
    }
}

void require_bf16(const Tensor& t, const char* label) {
    if (t.dtype != DType::BF16) {
        throw std::invalid_argument(std::string("ple_mix: ") + label + " must be BF16");
    }
}

void require_f32(const Tensor& t, const char* label) {
    if (t.dtype != DType::FP32) {
        throw std::invalid_argument(std::string("ple_mix: ") + label + " must be FP32");
    }
}

void require_rows_tokens(const Tensor& t, std::int32_t rows, std::int32_t tokens,
                         const char* label) {
    if (t.ne[0] != rows || t.ne[1] != tokens || t.ne[2] != 1 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string("ple_mix: ") + label +
                                    " must be contiguous with the expected shape");
    }
}
} // namespace

void ple_stream_gate(const Tensor& key, const Tensor& query, Tensor& out, cudaStream_t stream) {
    require_bf16(key, "key");
    require_bf16(query, "query");
    require_f32(out, "out");
    const Shape2 xy = require_2d(key, "key");
    require_same2(key, query, "query");
    if (out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() || out.data == nullptr ||
        out.ne[1] != xy.tokens) {
        throw std::invalid_argument("ple_stream_gate: out must be FP32 [hc, T]");
    }
    const std::int32_t hc = static_cast<std::int32_t>(out.ne[0]);
    if (hc < 1 || xy.rows % hc != 0) {
        throw std::invalid_argument("ple_stream_gate: out rows (hc) must divide key rows");
    }
    detail::ple_stream_gate_launch(key, query, out, stream);
}

void ple_gate_scale(const Tensor& value, const Tensor& gate, Tensor& out, cudaStream_t stream) {
    require_bf16(value, "value");
    require_f32(gate, "gate");
    require_bf16(out, "out");
    const Shape2 xy = require_2d(value, "value");
    if (out.ne[0] < xy.rows || out.ne[0] % xy.rows != 0) {
        throw std::invalid_argument("ple_gate_scale: out rows must be a multiple of value rows");
    }
    const std::int32_t hc = static_cast<std::int32_t>(out.ne[0]) / xy.rows;
    require_rows_tokens(gate, hc, xy.tokens, "gate");
    require_rows_tokens(out, xy.rows * hc, xy.tokens, "out");
    detail::ple_gate_scale_launch(value, gate, out, stream);
}

void dilated_causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                Tensor& out, cudaStream_t stream) {
    dilated_causal_conv1d_silu(x, weight, conv_state, conv_state, out, stream);
}

void dilated_causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                Tensor& conv_state_out, Tensor& out, cudaStream_t stream) {
    require_bf16(x, "x");
    require_bf16(weight, "weight");
    require_bf16(conv_state_in, "conv_state_in");
    require_bf16(conv_state_out, "conv_state_out");
    require_bf16(out, "out");
    const Shape2 xy = require_2d(x, "x");
    const std::int32_t C = xy.rows;
    if (weight.ne[0] != 4 || weight.ne[1] != C || weight.ne[2] != 1 || weight.ne[3] != 1 ||
        !weight.is_contiguous() || weight.data == nullptr) {
        throw std::invalid_argument("dilated_causal_conv1d_silu: weight must be BF16 [4, C]");
    }
    const auto require_state = [&](const Tensor& t, const char* label) {
        if (t.ne[0] != C || t.ne[1] != 9 || t.ne[2] != 1 || t.ne[3] != 1 || !t.is_contiguous() ||
            t.data == nullptr) {
            throw std::invalid_argument(std::string("dilated_causal_conv1d_silu: ") + label +
                                        " must be BF16 [C, 9]");
        }
    };
    require_state(conv_state_in, "conv_state_in");
    require_state(conv_state_out, "conv_state_out");
    require_rows_tokens(out, C, xy.tokens, "out");
    detail::dilated_causal_conv1d_silu_launch(x, weight, conv_state_in, conv_state_out, out,
                                              stream);
}

} // namespace ninfer::ops
