#pragma once

#include "ops/linear/bf16/bf16_gemm_mma.cuh"

namespace ninfer::ops::detail {

// Measured large-T production schedule for the BF16 computation core. Geometry remains a template
// argument so an exact problem can replace any tile, pipeline, cache, raster, or fragment choice
// without changing either Linear or semantic-Op dispatch.
template <class Geometry>
struct Bf16MmaProductionScheduleSelector {
    using Type =
        Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
};

// qwen4exp GDN a/b projection [96, 2560]: M=96 is not a multiple of the 64-row block; a 32-row
// block covers it exactly (3 tiles) with the same K tile and pipeline.
template <>
struct Bf16MmaProductionScheduleSelector<Bf16GemvGeometry<96, 2560>> {
    using Type =
        Bf16MmaSchedule<32, 128, 64, 16, 32, 2, 2, Cache::cg, Cache::cg,
                        Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
};

template <class Geometry>
using Bf16MmaProductionSchedule = typename Bf16MmaProductionScheduleSelector<Geometry>::Type;

} // namespace ninfer::ops::detail
