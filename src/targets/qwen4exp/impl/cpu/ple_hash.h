#pragma once

// PLE n-gram hash arithmetic (K0, layer 1 only) — pure integer math, no CUDA, no I/O.
//
// Direct port of the row-index loop of `llm_graph_input_ple::set_input`
// (llama.cpp @ cc83d7b48, src/models/qwen4exp.cpp:990). Geometry is fixed for
// the qwen3.8-flash-next artifact: ngram_size 3, heads_per_ngram 8, 16 heads
// (heads 0..7: 2-gram, heads 8..15: 3-gram), 160-dim rows.
//
// Semantics (reference, verbatim):
//   ctx[0] = the current token (its own EOS does not cut its own context);
//   for s = 1..ngram-1: t = predecessor s positions back;
//       cut  = cut || (t missing) || (t == eos);
//       ctx[s] = cut ? eos : t;
//   i.e. once a predecessor is missing (before sequence start) or EOS, every
//   further predecessor slot is EOS. Missing predecessors pad with EOS — not 0.
//   n = 2: mixed = u64(ctx[0])*m0 ^ u64(ctx[1])*m1          -> heads 0..7
//   n = 3: mixed ^= u64(ctx[2])*m2                          -> heads 8..15
//   row_h = mixed % size_h + offset_h  (u64 math).
//
// The reference's image-token branch (ubatch->token null -> hash with the
// image token id) is unreachable on this target: v1 is text-only, every
// position always carries a token id.

#include "targets/qwen4exp/impl/config.h"

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen4exp::detail {

inline constexpr std::uint32_t ple_ngram_size      = 3;
inline constexpr std::uint32_t ple_heads_per_ngram = 8;
inline constexpr std::uint32_t ple_n_heads         = 16; // (3-1) * 8

// Computes the 16 PLE table row indices for position `i` of the token sequence `ids`.
// `multipliers` / `head_offsets` / `head_vocab_sizes` are the artifact descriptor values
// (text/ple/layer_multipliers, head_offsets, head_vocab_sizes; u64 values lifted from the
// stored I32 lo/hi pairs). All arithmetic is u64: the products wrap modulo 2^64 (well-defined
// unsigned overflow), matching the reference's `uint64_t` multiplication.
inline std::array<std::uint64_t, ple_n_heads> compute_ple_rows(
    std::int32_t i, const std::int32_t* ids,
    const std::array<std::uint64_t, 3>& multipliers, const std::array<std::uint64_t, ple_n_heads>& head_offsets,
    const std::array<std::uint64_t, ple_n_heads>& head_vocab_sizes, std::int64_t eos) {
    // Predecessor context with the EOS cut.
    std::int64_t ctx[ple_ngram_size] = {};
    ctx[0] = (i >= 0) ? static_cast<std::int64_t>(ids[i]) : eos;
    bool cut = false;
    for (std::uint32_t s = 1; s < ple_ngram_size; ++s) {
        const bool missing = (i - static_cast<std::int32_t>(s)) < 0;
        const std::int64_t t =
            (cut || missing) ? -1 : static_cast<std::int64_t>(ids[i - static_cast<std::int32_t>(s)]);
        cut = cut || (t < 0) || (t == eos);
        ctx[s] = cut ? eos : t;
    }

    std::array<std::uint64_t, ple_n_heads> rows{};
    // n = 2 (heads 0..7): mixed over the two-gram.
    std::uint64_t mixed = static_cast<std::uint64_t>(ctx[0]) * multipliers[0] ^
                          static_cast<std::uint64_t>(ctx[1]) * multipliers[1];
    for (std::uint32_t h = 0; h < ple_heads_per_ngram; ++h) {
        rows[h] = mixed % head_vocab_sizes[h] + head_offsets[h];
    }
    // n = 3 (heads 8..15): extend the same mixed value with the third context token.
    mixed ^= static_cast<std::uint64_t>(ctx[2]) * multipliers[2];
    for (std::uint32_t h = ple_heads_per_ngram; h < ple_n_heads; ++h) {
        rows[h] = mixed % head_vocab_sizes[h] + head_offsets[h];
    }
    return rows;
}

} // namespace ninfer::targets::qwen4exp::detail
