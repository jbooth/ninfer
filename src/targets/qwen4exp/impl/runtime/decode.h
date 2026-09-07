#pragma once

// qwen4exp (jbq4) decode dataflow (JM4b): the 48-layer eager, text-only, no-CUDA-graph forward.
//
// The Program owns the persistent per-lane device state (GDN SSM/conv, QSA main KV + side cache,
// PLE dilated-conv history) and the load-time transforms (GDN A_log inversion, conv
// transpose+cast, FP32->BF16 norm caches). `run_decode_round` executes one round for B active
// lanes: each lane processes its current token at its own position, appends its per-lane KV/conv
// state, and produces one sampled token. Fully eager; the target owns the orchestration and the
// layout glue (GDN qkv assembly, QSA head/token transpose).

#include <ninfer/targets/qwen4exp/model_view.h>
#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {

// One active lane for a decode round: its state index (the lane id), its full on-host sequence
// (prompt + committed generated tokens), and the 0-based position of the token being processed
// (= sequence.size() - 1). The QSA KV for this token is appended at `position`, so the live prefix
// after the append is `position + 1`.
struct RoundLane {
    std::uint32_t lane;
    std::span<const TokenId> sequence;
    std::int32_t position;
};

// Persistent per-lane device state + round scratch for the decode dataflow. Allocated and zeroed
// in the Program constructor; the per-lane buffers are indexed by the lane id in `RoundLane`.
//
// Buffer layouts (all [layer][lane][...], lane = state slot):
//   gdn_ssm        FP32  [36 * C * (128*128*48)]   -- per (gdn layer, lane) 128x128x48 matrix.
//   gdn_conv_state BF16  [36 * C * (10240*3)]      -- per (gdn layer, lane) [10240, 3] window.
//   gdn_conv_w     BF16  [36 * (10240*4)]          -- per gdn layer [10240, 4] (transpose+cast).
//   gdn_a_log      FP32  [36 * 48]                 -- per gdn layer A_raw (inverted).
//   gdn_dt_bias    FP32  [36 * 48]                 -- per gdn layer dt_bias (raw).
//   ple_conv_hist  BF16  [C * (10240*9)]           -- per lane [10240, 9] dilated-conv history.
//   qsa_kv_k       BF16  [12 * C * kv_cap * 512]   -- per (qsa layer, lane) [256, 2, kv_cap].
//   qsa_kv_v       BF16  [12 * C * kv_cap * 512]   -- per (qsa layer, lane) [256, 2, kv_cap].
//   qsa_side       BF16  [12 * C * kv_cap * 128]   -- per (qsa layer, lane) [128, kv_cap].
//   norm_cache     BF16  [36*128 + 12*(256+256+128+128)] -- GDN [128] then QSA norms.
//   scratch        (round scratch, carved with a WorkspaceArena).
struct DecodeState {
    std::int32_t max_concurrency = 0;
    std::int32_t kv_capacity     = 0;

    DeviceBuffer gdn_ssm;
    DeviceBuffer gdn_conv_state;
    DeviceBuffer gdn_conv_w;
    DeviceBuffer gdn_a_log;
    DeviceBuffer gdn_dt_bias;

    DeviceBuffer ple_conv_hist;

    DeviceBuffer qsa_kv_k;
    DeviceBuffer qsa_kv_v;
    DeviceBuffer qsa_side;

    DeviceBuffer norm_cache;
    DeviceBuffer scratch;
};

// Runs the 48-layer decode/prefill dataflow for the B lanes in `round` and returns one sampled
// token id per lane. `model` carries the device weight views; `host` the page-cache host view
// (MoE banks, token embedding, PLE); `state` the persistent per-lane state + scratch.
std::vector<TokenId> run_decode_round(DecodeState& state, const ModelView& model,
                                      const HostArtifactView& host, cudaStream_t stream,
                                      const std::span<const RoundLane>& round);

// Applies the load-time transforms (GDN A_log inversion, conv transpose+cast, FP32->BF16 norm
// caches) into `state` from the raw device weights in `model`. Called once after the state buffers
// are allocated and zeroed.
void apply_load_time_transforms(DecodeState& state, const ModelView& model, cudaStream_t stream);

} // namespace ninfer::targets::qwen4exp::detail
