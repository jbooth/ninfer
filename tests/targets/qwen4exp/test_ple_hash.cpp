// M1 gate (phase3.md §4.4): bit-exact parity of the qwen4exp PLE n-gram hash math
// (`compute_ple_rows`, src/targets/qwen4exp/impl/cpu/ple_hash.h) against an
// independent reimplementation of `llm_graph_input_ple::set_input`
// (llama.cpp @ cc83d7b48, src/models/qwen4exp.cpp:990).
//
// The test has three parts, all host-only (no CUDA):
//   1. Parity of `compute_ple_rows` against an inline reference written in a
//      different style, over synthetic descriptors (small prime sub-ranges) and
//      the real artifact descriptors (large multipliers -> u64 wrap). Sequences
//      cover: sequence start (missing predecessors), a token's own EOS, an EOS
//      predecessor cut, normal mid-stream, and LCG-random sequences.
//   2. Recorded oracle values: the 16 row indices per position of a fixed
//      6-token sequence, computed by the independent Python oracle
//      (tools/parity/qwen4exp/ple_reference.py --descriptors ARTIFACT) and
//      recorded here verbatim. This pins the arithmetic end-to-end without
//      needing the artifact or Python at test time.
//   3. Artifact mode (argv[1]): loads the three descriptor objects through
//      the real artifact Reader, asserts they equal the embedded literals,
//      recomputes the recorded rows, checks table-range validity, and preads
//      the 320-byte table row at each computed offset (the gather byte path).
//      Exits 77 (skip) when no artifact path is supplied.

#include "artifact/reader.h"
#include "targets/qwen4exp/impl/cpu/ple_hash.h"

#include <array>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using rows_t = std::array<std::uint64_t, ninfer::targets::qwen4exp::detail::ple_n_heads>;

using ninfer::targets::qwen4exp::detail::compute_ple_rows;

template <std::size_t N>
std::array<std::uint64_t, N> arr_of(const std::uint64_t* p) {
    std::array<std::uint64_t, N> a{};
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = p[i];
    }
    return a;
}

// Independent inline reference, written directly from the llama.cpp reference loop
// (qwen4exp.cpp:990-1040), in a deliberately different structure from the SUT.
rows_t reference_rows(const std::vector<std::int32_t>& ids, std::size_t pos, std::int64_t eos,
                      const std::array<std::uint64_t, 3>& mult,
                      const std::array<std::uint64_t, 16>& offs, const std::array<std::uint64_t, 16>& sz) {
    // Context window [current, prev1, prev2] with the EOS cut.
    std::int64_t ctx[3];
    ctx[0] = static_cast<std::int64_t>(ids[pos]);
    bool cut = false;
    for (std::uint32_t s = 1; s <= 2; ++s) {
        const bool gone = cut || pos < s;
        const std::int64_t t = gone ? -1 : static_cast<std::int64_t>(ids[pos - s]);
        cut = cut || (t < 0) || (t == eos);
        ctx[s] = cut ? eos : t;
    }
    rows_t out{};
    // Heads 0..7: two-gram hash; heads 8..15: three-gram hash (the same mixed
    // value extended with the third context token).
    std::uint64_t mix = static_cast<std::uint64_t>(ctx[0]) * mult[0] ^
                        static_cast<std::uint64_t>(ctx[1]) * mult[1];
    for (std::uint32_t h = 0; h < 8; ++h) {
        out[h] = mix % sz[h] + offs[h];
    }
    mix ^= static_cast<std::uint64_t>(ctx[2]) * mult[2];
    for (std::uint32_t h = 8; h < 16; ++h) {
        out[h] = mix % sz[h] + offs[h];
    }
    return out;
}

int check_parity(const char* label, const std::vector<std::int32_t>& ids,
                 const std::array<std::uint64_t, 3>& mult, const std::array<std::uint64_t, 16>& offs,
                 const std::array<std::uint64_t, 16>& sz, std::int64_t eos) {
    for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        const auto a = compute_ple_rows(static_cast<std::int32_t>(pos), ids.data(), mult, offs, sz, eos);
        const auto b = reference_rows(ids, pos, eos, mult, offs, sz);
        for (std::uint32_t h = 0; h < 16; ++h) {
            if (a[h] != b[h]) {
                std::cerr << "PARITY MISMATCH " << label << " pos=" << pos << " head=" << h
                          << " got=" << a[h] << " want=" << b[h] << "\n";
                return 1;
            }
        }
    }
    return 0;
}

// ---- Real artifact descriptors (recorded 2026-08-31; the artifact objects were
// patched to these values the same day, see stage1.md). ----
const std::uint64_t REAL_MULT[3] = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
const std::uint64_t REAL_OFFS[16] = {
    0ULL,        20000003ULL, 40000026ULL, 60000059ULL, 80000106ULL, 100000165ULL,
    120000228ULL, 140000297ULL, 160000374ULL, 180000455ULL, 200000548ULL, 220000655ULL,
    240000802ULL, 260000955ULL, 280001114ULL, 300001275ULL};
const std::uint64_t REAL_SZ[16] = {
    20000003ULL, 20000023ULL, 20000033ULL, 20000047ULL, 20000059ULL, 20000063ULL,
    20000069ULL, 20000077ULL, 20000081ULL, 20000093ULL, 20000107ULL, 20000147ULL,
    20000153ULL, 20000159ULL, 20000161ULL, 20000171ULL};
const std::int64_t EOS = 248044;

// Fixed recorded sequence: 5, EOS, 7, 11, 13, 17 (own-EOS at pos 1; EOS predecessor
// cut at pos 3; missing predecessors at pos 0/1).
const std::int32_t REC_IDS[] = {5, 248044, 7, 11, 13, 17};
// Recorded by tools/parity/qwen4exp/ple_reference.py --descriptors <artifact> on
// 2026-08-31, 16 rows x 6 positions, verbatim.
const std::uint64_t REC_ROWS[6][16] = {
    {15389869ULL, 39778609ULL, 55713969ULL, 62213332ULL, 88817728ULL, 118483999ULL,
     133731511ULL, 155458159ULL, 179763390ULL, 197956758ULL, 205378969ULL, 220499474ULL,
     242466248ULL, 265658744ULL, 293662119ULL, 315720898ULL},
    {18392917ULL, 34857659ULL, 47499630ULL, 74137148ULL, 90126688ULL, 103063933ULL,
     133351754ULL, 142048315ULL, 173113148ULL, 192355065ULL, 202997842ULL, 236534513ULL,
     246250533ULL, 276928146ULL, 287367671ULL, 301167364ULL},
    {2927653ULL, 34980843ULL, 54748278ULL, 66612378ULL, 97814964ULL, 109013870ULL,
     126560393ULL, 151352333ULL, 167888935ULL, 182235580ULL, 215170017ULL, 237467519ULL,
     247510681ULL, 278779700ULL, 296141806ULL, 304994522ULL},
    {3518994ULL, 23525975ULL, 43529577ULL, 79534759ULL, 87539255ULL, 103540813ULL,
     127543174ULL, 159546370ULL, 168682726ULL, 193417972ULL, 204093272ULL, 222522584ULL,
     258665398ULL, 275167591ULL, 280748216ULL, 309250551ULL},
    {9783552ULL, 38030433ULL, 42154252ULL, 63928059ULL, 96877490ULL, 114527408ULL,
     121002300ULL, 156302400ULL, 175668638ULL, 197581371ULL, 206479972ULL, 226193110ULL,
     247150585ULL, 268108162ULL, 295094204ULL, 310023618ULL},
    {8107993ULL, 21774939ULL, 48608765ULL, 70176473ULL, 94377727ULL, 109111570ULL,
     121212369ULL, 150680275ULL, 176419043ULL, 188217223ULL, 208648940ULL, 221313544ULL,
     247213771ULL, 273114100ULL, 281747565ULL, 304914889ULL}};

// Lift the stored I32 lo/hi pairs to u64 (little-endian, lo first).
void lift(const std::span<const std::byte>& raw, std::vector<std::uint64_t>& out) {
    out.clear();
    for (std::size_t i = 0; i + 8 <= raw.size(); i += 8) {
        std::uint64_t v = 0;
        for (int b = 7; b >= 0; --b) {
            v = (v << 8) | static_cast<std::uint64_t>(raw[i + std::size_t(b)]);
        }
        out.push_back(v);
    }
}

int artifact_mode(const std::string& path) {
    ninfer::artifact::Reader reader(path);
    const std::array<const char*, 3> names = {"text/ple/layer_multipliers",
                                             "text/ple/head_offsets",
                                             "text/ple/head_vocab_sizes"};
    std::vector<std::vector<std::uint64_t>> lifted(3);
    for (std::size_t k = 0; k < 3; ++k) {
        const auto* obj = reader.find(names[k]);
        if (obj == nullptr) {
            std::cerr << "missing artifact object " << names[k] << "\n";
            return 1;
        }
        const auto span = reader.payload(*obj);
        lift({reinterpret_cast<const std::byte*>(span.data.data()), span.data.size()}, lifted[k]);
        if (lifted[k].size() != (k == 0 ? 3 : 16)) {
            std::cerr << "unexpected lifted count for " << names[k] << ": " << lifted[k].size()
                      << "\n";
            return 1;
        }
    }
    const std::uint64_t* want[3] = {REAL_MULT, REAL_OFFS, REAL_SZ};
    const std::size_t counts[3] = {3, 16, 16};
    for (std::size_t k = 0; k < 3; ++k) {
        for (std::size_t i = 0; i < counts[k]; ++i) {
            if (lifted[k][i] != want[k][i]) {
                std::cerr << "artifact descriptor drift " << names[k] << "[" << i << "]: got "
                          << lifted[k][i] << " want " << want[k][i] << "\n";
                return 1;
            }
        }
    }

    // Table extents.
    const auto* table = reader.find("text/per_layer_token_embedding");
    if (table == nullptr) {
        std::cerr << "missing PLE table object\n";
        return 1;
    }
    const auto& td = std::get<ninfer::artifact::TensorDescriptor>(*table);
    const std::uint64_t rows = td.shape.front();
    const std::uint64_t row_bytes = td.shape.back() * 2U; // BF16
    if (rows != 320001536ULL || row_bytes != 320U) {
        std::cerr << "unexpected PLE table shape: " << rows << " x " << row_bytes / 2 << "\n";
        return 1;
    }

    // Recompute the recorded rows; assert in-range and pread the table row bytes.
    // Plain pread (O_DIRECT read_direct needs 4096 B alignment; rows are 320 B).
    std::vector<std::byte> host(320);
    const int row_fd = ::open(path.c_str(), O_RDONLY);
    if (row_fd < 0) {
        std::cerr << "cannot open artifact for row pread\n";
        return 1;
    }
    const std::vector<std::int32_t> ids(REC_IDS, REC_IDS + 6);
    for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        const auto got = compute_ple_rows(static_cast<std::int32_t>(pos), ids.data(),
                                          arr_of<3>(REAL_MULT), arr_of<16>(REAL_OFFS),
                                          arr_of<16>(REAL_SZ), EOS);
        for (std::uint32_t h = 0; h < 16; ++h) {
            if (got[h] != REC_ROWS[pos][h]) {
                std::cerr << "recorded mismatch pos=" << pos << " head=" << h << " got=" << got[h]
                          << " want=" << REC_ROWS[pos][h] << "\n";
                ::close(row_fd);
                return 1;
            }
            if (got[h] >= rows) {
                std::cerr << "row out of table range pos=" << pos << " head=" << h << " row="
                          << got[h] << "\n";
                ::close(row_fd);
                return 1;
            }
            const std::uint64_t off = td.offset + got[h] * row_bytes;
            const std::size_t n = ::pread(row_fd, host.data(), host.size(), off_t(off));
            if (n != host.size()) {
                std::cerr << "pread failed at pos=" << pos << " head=" << h << "\n";
                ::close(row_fd);
                return 1;
            }
        }
    }
    ::close(row_fd);
    std::cout << "artifact mode: descriptors match, " << ids.size() * 16 << " rows in range, "
              << "table rows pread ok\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    // Part 1: parity over synthetic + real descriptors.
    const std::uint64_t SYN_MULT[3] = {9999999999997ULL, 9999999999999ULL, 9999999999973ULL};
    // 16 sub-range sizes (~20000 each) with contiguous offsets from 0.
    const std::uint64_t SYN_SZ[16] = {20003ULL, 20011ULL, 20021ULL, 20029ULL, 20039ULL,
                                      20047ULL, 20051ULL, 20059ULL, 20063ULL, 20071ULL,
                                      20089ULL, 20101ULL, 20107ULL, 20113ULL, 20117ULL,
                                      20123ULL};
    std::uint64_t SYN_OFFS[16] = {};
    for (std::uint32_t h = 1; h < 16; ++h) {
        SYN_OFFS[h] = SYN_OFFS[h - 1] + SYN_SZ[h - 1];
    }
    const std::vector<std::int32_t> A = {5, 248044, 7, 11, 13, 17}; // recorded battery
    const std::vector<std::int32_t> B = {248044};                   // single EOS token
    const std::vector<std::int32_t> C = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<std::int32_t> D = {248044, 248044, 248044, 9}; // all-EOS prefix
    // LCG-random sequence, 64 tokens, EOS sprinkles, large ids (wrap forcing).
    const std::vector<std::int32_t> R = [] {
        std::uint64_t s = 0x9E3779B97F4A7C15ULL;
        std::vector<std::int32_t> v;
        for (int i = 0; i < 64; ++i) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            const std::uint32_t r = static_cast<std::uint32_t>(s >> 32);
            v.push_back(i % 13 == 0 ? 248044 : static_cast<std::int32_t>(r % 248320));
        }
        return v;
    }();
    const auto m_s = arr_of<3>(SYN_MULT);
    const auto o_s = arr_of<16>(SYN_OFFS);
    const auto z_s = arr_of<16>(SYN_SZ);
    const auto m_r = arr_of<3>(REAL_MULT);
    const auto o_r = arr_of<16>(REAL_OFFS);
    const auto z_r = arr_of<16>(REAL_SZ);
    if (check_parity("synthetic", A, m_s, o_s, z_s, EOS) ||
        check_parity("synthetic", B, m_s, o_s, z_s, EOS) ||
        check_parity("synthetic", C, m_s, o_s, z_s, EOS) ||
        check_parity("synthetic", D, m_s, o_s, z_s, EOS) ||
        check_parity("synthetic", R, m_s, o_s, z_s, EOS) ||
        check_parity("real", A, m_r, o_r, z_r, EOS) ||
        check_parity("real", R, m_r, o_r, z_r, EOS)) {
        return 1;
    }
    std::cout << "parity ok: 7 descriptor x sequence batteries\n";

    // Part 2: recorded oracle values (real descriptors, fixed sequence).
    const std::vector<std::int32_t> ids(REC_IDS, REC_IDS + 6);
    for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        const auto got = compute_ple_rows(static_cast<std::int32_t>(pos), ids.data(), m_r, o_r, z_r,
                                          EOS);
        for (std::uint32_t h = 0; h < 16; ++h) {
            if (got[h] != REC_ROWS[pos][h]) {
                std::cerr << "recorded mismatch pos=" << pos << " head=" << h << " got=" << got[h]
                          << " want=" << REC_ROWS[pos][h] << "\n";
                return 1;
            }
        }
    }
    std::cout << "recorded rows ok: 6 positions x 16 heads\n";
    std::cout << "M1 PLE hash gate passed\n";

    // Part 3: artifact cross-check when a path is supplied (the recorded values above
    // are the gate; this additionally verifies the live artifact holds them).
    if (argc >= 2) {
        return artifact_mode(argv[1]);
    }
    std::cout << "(no artifact path supplied; skipped the live-artifact cross-check)\n";
    return 0;
}
