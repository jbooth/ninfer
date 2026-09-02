// QSA q|gate de-interleave (kernel N) qualification.
//
// The op transposes the q block [0,6144) and gate block [6656,12800) of the attn_input_proj
// parent [13312,T] (q | k | gate | v, head-major: head h / dim d at row h*256+d) into the
// dim-major [256,24,T] layout. The transformation is a pure data move (no arithmetic), so the
// BF16 bits must be reproduced exactly; the oracle recomputes the expected layout on the host
// and compares bit-for-bit.

#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kParentRows = 13312;
constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kQOffset    = 0;
constexpr std::int32_t kGateOffset = 6656;

struct Case {
    std::int32_t tokens;
    std::uint32_t seed;
};

std::int32_t run_case(const Case& c, std::mt19937& rng) {
    const std::int32_t tokens = c.tokens;

    // Represented BF16 parent (row-major [13312, T]).
    std::vector<float> parent(static_cast<std::size_t>(kParentRows) * tokens);
    fill_uniform(parent, c.seed, -2.0F, 2.0F);
    round_to_bf16(parent);

    // Expected dim-major layout, recomputed on the host (bit-exact source of the move).
    std::vector<float> q_ref(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    std::vector<float> gate_ref(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t h = 0; h < kQueryHeads; ++h) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::int64_t row   = static_cast<std::int64_t>(h) * kHeadDim + d;
                const std::size_t out    =
                    static_cast<std::size_t>(d) * kQueryHeads * tokens +
                    static_cast<std::size_t>(h) * tokens + t;
                q_ref[out]    = parent[(kQOffset + row) * tokens + t];
                gate_ref[out] = parent[(kGateOffset + row) * tokens + t];
            }
        }
    }

    // Device inputs + run.
    const DeviceBuffer dparent = to_device_bf16(parent),
        dq(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * 2),
        dgate(static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens * 2);
    const Tensor tparent(dparent.p, DType::BF16, {kParentRows, tokens});
    Tensor tq(dq.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor tgate(dgate.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    ops::qsa_split_qgate(tparent, tq, tgate, 0);
    cuda_synchronize();

    const std::vector<double> got_q =
        from_device_bf16(dq, static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);
    const std::vector<double> got_gate =
        from_device_bf16(dgate, static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens);

    // Bit-exact: a pure data move must reproduce the represented BF16 values exactly.
    int bad = 0;
    for (std::size_t i = 0; i < got_q.size(); ++i) {
        if (got_q[i] != double(q_ref[i])) {
            if (bad < 4) std::cerr << "  q [" << i << "] got " << got_q[i] << " ref " << q_ref[i]
                                   << "\n";
            ++bad;
        }
    }
    for (std::size_t i = 0; i < got_gate.size(); ++i) {
        if (got_gate[i] != double(gate_ref[i])) {
            if (bad < 4) std::cerr << "  gate [" << i << "] got " << got_gate[i] << " ref "
                                   << gate_ref[i] << "\n";
            ++bad;
        }
    }
    if (bad) {
        std::cerr << "qsa_split T=" << tokens << " seed=" << c.seed << ": " << bad
                  << " mismatched element(s)\n";
        return 1;
    }
    return 0;
}
} // namespace

int main() {
    if (cuda_unavailable()) {
        std::printf("no CUDA device; skipping\n");
        return 77;
    }
    std::mt19937 rng(0x2b7);
    const Case cases[] = {
        {1, 201}, {2, 202}, {3, 203}, {5, 205}, {7, 207}, {8, 208},
    };
    int failures = 0;
    for (const Case& c : cases) failures += run_case(c, rng);

    // Negative: the op must reject a parent with the wrong row extent.
    {
        DeviceBuffer p(static_cast<std::size_t>(13313) * 2 * 2);
        Tensor bad_p(p.p, DType::BF16, {13313, 2});
        DeviceBuffer o(static_cast<std::size_t>(kHeadDim) * kQueryHeads * 2 * 2);
        Tensor oq(o.p, DType::BF16, {kHeadDim, kQueryHeads, 2});
        Tensor og(o.p, DType::BF16, {kHeadDim, kQueryHeads, 2});
        try {
            ops::qsa_split_qgate(bad_p, oq, og, 0);
            std::cerr << "qsa_split: parent row extent should be rejected\n";
            ++failures;
        } catch (const std::exception&) {
        }
    }

    if (failures) {
        std::cerr << failures << " qsa_split check(s) failed\n";
        return 1;
    }
    std::printf("qsa_split: all checks passed\n");
    return 0;
}
