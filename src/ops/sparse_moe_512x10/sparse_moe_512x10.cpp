#include "ninfer/ops/sparse_moe_512x10.h"

#include "ops/sparse_moe_512x10/sparse_moe_512x10_launch.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t align256(std::int64_t v) { return (v + 255) & ~std::int64_t{255}; }

} // namespace

std::size_t sparse_moe_512x10_workspace_capacity_bytes(const SparseMoe512x10Geometry& geo,
                                                       std::int32_t T) {
    std::int64_t off = 0;
    off += align256(std::int64_t(geo.n_experts + 1) * T * 4);            // scores F32
    off += align256(std::int64_t(geo.top_k) * T * 4);                    // topk_ids I32
    off += align256(std::int64_t(geo.top_k) * T * 4);                    // topk_w F32
    off += align256(std::int64_t(T) * 4);                                // shared_logit F32
    off += align256(std::int64_t(geo.top_k) * geo.inter * T * 2);        // act BF16
    off += align256(std::int64_t(geo.top_k) * geo.hidden * T * 4);       // y F32
    off += align256(std::int64_t(geo.inter) * T * 2);                    // shared_act BF16
    off += align256(std::int64_t(geo.hidden) * T * 4);                   // shared_y F32
    return static_cast<std::size_t>(off);
}

void sparse_moe_512x10(const Tensor& x, const SparseMoe512x10Geometry& geo,
                       const SparseMoe512x10Weights& w, SparseMoe512x10Epilogue epilogue,
                       Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || destination.dtype != DType::BF16 ||
        x.ne[0] <= 0 || x.ne[1] < 1 || x.ne[2] != 1 || x.ne[3] != 1 || !x.is_contiguous() ||
        x.data == nullptr || destination.ne[0] <= 0 || destination.ne[1] < 1 ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("sparse_moe_512x10: x/destination must be contiguous "
                                    "BF16 [hidden, T]");
    }
    if (geo.n_experts <= 0 || geo.top_k <= 0 || geo.top_k > 16 || geo.hidden != 2560 ||
        geo.inter != 640 || geo.hidden % 32 != 0 || geo.inter % 64 != 0) {
        throw std::invalid_argument("sparse_moe_512x10: unsupported geometry (hidden=2560, "
                                    "inter=640, top_k<=16)");
    }
    const std::int32_t H = x.ne[0];
    const std::int32_t T = x.ne[1];
    if (H != geo.hidden || destination.ne[0] != H || destination.ne[1] != T) {
        throw std::invalid_argument("sparse_moe_512x10: x/destination shape mismatch");
    }
    if (epilogue != SparseMoe512x10Epilogue::AddResidual) {
        throw std::invalid_argument("sparse_moe_512x10: unsupported epilogue");
    }

    const std::size_t need = sparse_moe_512x10_workspace_capacity_bytes(geo, T);
    std::uint8_t* p = static_cast<std::uint8_t*>(workspace.alloc_bytes(need).data);
    if (p == nullptr) {
        throw std::runtime_error("sparse_moe_512x10: workspace allocation failed");
    }
    std::int64_t off = 0;
    auto carve = [&](std::int64_t bytes) -> void* {
        void* out = p;
        off += align256(bytes);
        p += align256(bytes);
        return out;
    };
    float* scores      = static_cast<float*>(carve(std::int64_t(geo.n_experts + 1) * T * 4));
    int* topk_ids      = static_cast<int*>(carve(std::int64_t(geo.top_k) * T * 4));
    float* topk_w      = static_cast<float*>(carve(std::int64_t(geo.top_k) * T * 4));
    float* shared_logit = static_cast<float*>(carve(std::int64_t(T) * 4));
    __nv_bfloat16* act = static_cast<__nv_bfloat16*>(
        carve(std::int64_t(geo.top_k) * geo.inter * T * 2));
    float* y        = static_cast<float*>(carve(std::int64_t(geo.top_k) * geo.hidden * T * 4));
    __nv_bfloat16* s_act = static_cast<__nv_bfloat16*>(carve(std::int64_t(geo.inter) * T * 2));
    float* s_y      = static_cast<float*>(carve(std::int64_t(geo.hidden) * T * 4));
    (void)off;

    sparse_moe_512x10_detail::run(
        static_cast<const __nv_bfloat16*>(x.data), geo.n_experts, geo.top_k, T, w.router,
        w.input_divisor, w.gate_up_codes, w.gate_up_scales, w.down_codes, w.down_high,
        w.down_scales, w.shared_gate_up_codes, w.shared_gate_up_scales, w.shared_down_codes,
        w.shared_down_scales, static_cast<__nv_bfloat16*>(destination.data), scores, topk_ids,
        topk_w, shared_logit, act, y, s_act, s_y, stream);
}

} // namespace ninfer::ops
