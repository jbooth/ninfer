// bench/ops/sparse_moe_512x10_cpu_bench.cpp
//
// DRAM-floor microbenchmark for the production CPU MoE op
// `ninfer::ops::sparse_moe_512x10_cpu` (decode route, T = 1) at the project's real
// topology: 512 routed experts, top-10, hidden 2560, inter 640, gs=64, FP16 scales —
// the jbq4 Q4G64 / Q8G64 banks. Randomized weights (fixed-seed LCG) in the exact
// packed bank layout the op consumes (4 separate contiguous planes: gu_base, gu_scales,
// dn_base, dn_scales, each spanning all E experts).
//
// Cold-cache discipline: before every timed run, every participating thread writes a
// 128 MiB bogus buffer (per-thread) to evict the op's touched lines from L2. The full
// bank is multi-GB, so it never sits in L2; the timed traffic is the page-cache read of
// the selected experts' planes plus the shared expert and router.
//
// Reported metrics (steady-state median across iterations):
//   - layer time        : wall time to finish the whole MoE layer (one op call, T=1)
//   - FLOP/s            : exact arithmetic of the closed op:
//       2*(513*H)                     router GEMVs (512 routed + shared row)
//     + (top_k + 1) * 2*(2*I*H + H*I) gate/up (2I x H) + down (H x I) per expert
//   - expert-GB/s       : bytes of expert weight planes actually read (the 10 selected
//       experts' gate_up+down base+scales + the shared expert's), divided by layer time.
//
// Usage: ninfer_sparse_moe_512x10_cpu_bench [--threads N] [--iters N] [--q8] [--no-cold]

#include "ninfer/ops/sparse_moe_512x10_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;

namespace {

constexpr std::int32_t kH = 2560;
constexpr std::int32_t kI = 640;
constexpr std::int32_t kE = 512;
constexpr std::int32_t kK = 10;
constexpr std::int64_t kG = kH / 64;  // gate groups
constexpr std::int64_t kD = kI / 64;  // down groups

// Per-expert plane bytes (base + scales for gate_up + down).
std::size_t expert_bytes(MoeCode code) {
    const std::int64_t gu_rows = 2 * kI;  // 2I rows of H codes
    const std::int64_t dn_rows = kH;      // H rows of I codes
    const std::size_t gu_base = code == MoeCode::Q4G64 ? gu_rows * kH / 2 : gu_rows * kH;
    const std::size_t dn_base = code == MoeCode::Q4G64 ? dn_rows * kI / 2 : dn_rows * kI;
    const std::size_t scales = (gu_rows * kG + dn_rows * kD) * 2;
    return gu_base + dn_base + scales;
}

// Exact closed-op FLOPs for one T=1 layer.
std::int64_t layer_flops(std::int32_t top_k) {
    const std::int64_t expert = 2LL * (2LL * kI * kH + std::int64_t(kH) * kI);
    return 2LL * std::int64_t(kE + 1) * kH + std::int64_t(top_k + 1) * expert;
}

// Deterministic per-element LCG (fixed global state: the bench is reproducible).
std::uint64_t g_rng = 0x9E3779B97F4A7C15ull;
std::uint64_t lcg_next() {
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return g_rng >> 16;
}

void init_u8(std::uint8_t* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = static_cast<std::uint8_t>(lcg_next());
    }
}

void init_fp16(std::uint16_t* p, std::size_t n) {
    // Small positive subnormal/normal FP16 scales (~1e-4), deterministic.
    for (std::size_t i = 0; i < n; ++i) {
        std::uint16_t m = static_cast<std::uint16_t>((lcg_next() >> 24) & 0x3FF);
        if (m == 0) m = 1;
        p[i] = static_cast<std::uint16_t>(m);
    }
}

void init_bf16(std::int16_t* p, std::size_t n, float scale) {
    for (std::size_t i = 0; i < n; ++i) {
        const double v = (static_cast<double>(lcg_next() % 1000000) / 500000.0 - 1.0) * scale;
        std::uint32_t bits;
        float f = static_cast<float>(v);
        std::memcpy(&bits, &f, 4);
        bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
        p[i] = static_cast<std::int16_t>(static_cast<std::uint16_t>(bits >> 16));
    }
}

void init_router(std::vector<float>& router) {
    for (int e = 0; e <= kE; ++e) {
        for (int c = 0; c < kH; ++c) {
            router[static_cast<std::size_t>(e) * kH + c] =
                static_cast<float>(static_cast<double>(lcg_next() % 2001 - 1000) / 100000.0);
        }
    }
    for (int e = 0; e < kE; ++e) {
        router[static_cast<std::size_t>(e) * kH] =
            static_cast<float>(static_cast<double>(lcg_next() % 4000 - 2000) / 200.0);
    }
}

// Per-thread bogus buffer for L2 eviction.
constexpr std::size_t kFlushBytes = 128ull << 20;

void flush_l2(int threads) {
    std::vector<std::vector<std::uint8_t>> buf(threads,
                                               std::vector<std::uint8_t>(kFlushBytes, 0xA5u));
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int h = 0; h < threads; ++h) {
        pool.emplace_back([&, h]() {
            std::uint8_t* p = buf[h].data();
            for (std::size_t i = 0; i < kFlushBytes; i += 64) {
                std::memcpy(p + i, "\x12\x34\x56\x78\x90\xAB\xCD\xEF", 8);
            }
        });
    }
    for (auto& t : pool) t.join();
}

struct Stats {
    std::vector<double> us;
    double min() const { return *std::min_element(us.begin(), us.end()); }
    double median() const {
        std::vector<double> v = us;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    double mean() const {
        double s = 0;
        for (double v : us) s += v;
        return s / us.size();
    }
};

void print_run(const char* tag, const Stats& s, double seconds_per, std::int64_t flops,
               std::size_t accessed, int threads) {
    const double med = s.median() * 1.0e-6;
    std::printf(
        "%-28s runs=%zu  median=%9.3f ms  min=%9.3f ms  mean=%9.3f ms\n"
        "    FLOP/s        : %10.3f GFLOP/s\n"
        "    expert GB/s   : %10.3f GB/s   (accessed %zu MiB = expert weights actually read)\n",
        tag, s.us.size(), s.median() * 1e-3, s.min() * 1e-3, s.mean() * 1e-3, flops / seconds_per / 1e9,
        static_cast<double>(accessed) / seconds_per / 1e9, accessed >> 20);
    (void)threads;
    (void)med;
}

}  // namespace

int main(int argc, char** argv) {
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 16;
    int iters = 8;
    MoeCode code = MoeCode::Q4G64;
    bool cold = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--q8") code = MoeCode::Q8G64;
        else if (a == "--no-cold") cold = false;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: %s [--threads N] [--iters N] [--q8] [--no-cold]\n", argv[0]);
            return 0;
        }
    }
    if (threads < 1) threads = 1;
    if (iters < 1) iters = 1;

    const std::size_t per_expert = expert_bytes(code);
    const std::size_t accessed = static_cast<std::size_t>(kK + 1) * per_expert;

    std::printf("sparse_moe_512x10_cpu bench: E=%d top_k=%d H=%d I=%d gs=64 %s, T=1, threads=%d, iters=%d\n",
                kE, kK, kH, kI, code == MoeCode::Q4G64 ? "Q4G64" : "Q8G64", threads, iters);
    std::printf("routed bank=%zu MiB resident; accessed/layer=%zu MiB; flops=%lld; op threads capped at T*(K+1)=%d\n",
                (static_cast<std::size_t>(kE) * per_expert) >> 20, accessed >> 20,
                static_cast<long long>(layer_flops(kK)), std::min(threads, kK + 1));

    // Routed: E experts. gu: E*2I rows of H codes, scales: E*2I*(H/64) fp16.
    //        dn: E*H rows of I codes, scales: E*H*(I/64) fp16.
    // Shared: 1 expert. Same shapes as one expert.
    auto alloc64 = [](std::size_t n) -> std::uint8_t* {
        std::uint8_t* p = static_cast<std::uint8_t*>(
            std::aligned_alloc(64, (n + 63) & ~static_cast<std::size_t>(63)));
        std::memset(p, 0, n);
        return p;
    };
    const std::int64_t rgu_rows = static_cast<std::int64_t>(kE) * 2 * kI;
    const std::int64_t rdn_rows = static_cast<std::int64_t>(kE) * kH;
    const std::size_t rgu_base_bytes = (code == MoeCode::Q4G64)
        ? static_cast<std::size_t>(rgu_rows) * kH / 2
        : static_cast<std::size_t>(rgu_rows) * kH;
    const std::size_t rgu_scales_bytes = static_cast<std::size_t>(rgu_rows) * kG * 2;
    const std::size_t rdn_base_bytes = (code == MoeCode::Q4G64)
        ? static_cast<std::size_t>(rdn_rows) * kI / 2
        : static_cast<std::size_t>(rdn_rows) * kI;
    const std::size_t rdn_scales_bytes = static_cast<std::size_t>(rdn_rows) * kD * 2;

    const std::size_t sgu_base_bytes = (code == MoeCode::Q4G64) ? 2 * kI * kH / 2 : 2 * kI * kH;
    const std::size_t sgu_scales_bytes = 2 * kI * kG * 2;
    const std::size_t sdn_base_bytes = (code == MoeCode::Q4G64) ? kH * kI / 2 : kH * kI;
    const std::size_t sdn_scales_bytes = kH * kD * 2;

    std::uint8_t* rgu_base = alloc64(rgu_base_bytes);
    std::uint8_t* rgu_scales = alloc64(rgu_scales_bytes);
    std::uint8_t* rdn_base = alloc64(rdn_base_bytes);
    std::uint8_t* rdn_scales = alloc64(rdn_scales_bytes);
    std::uint8_t* sgu_base = alloc64(sgu_base_bytes);
    std::uint8_t* sgu_scales = alloc64(sgu_scales_bytes);
    std::uint8_t* sdn_base = alloc64(sdn_base_bytes);
    std::uint8_t* sdn_scales = alloc64(sdn_scales_bytes);
    std::vector<float> router(static_cast<std::size_t>(kE + 1) * kH);
    std::vector<std::int16_t> x(kH);
    std::vector<std::int16_t> dest(kH);

    const std::size_t total_alloc = rgu_base_bytes + rgu_scales_bytes + rdn_base_bytes + rdn_scales_bytes
                                   + sgu_base_bytes + sgu_scales_bytes + sdn_base_bytes + sdn_scales_bytes;
    std::printf("randomizing bank (%zu MiB)...\n", total_alloc >> 20);
    init_u8(rgu_base, rgu_base_bytes);
    init_fp16(reinterpret_cast<std::uint16_t*>(rgu_scales), rgu_rows * kG);
    init_u8(rdn_base, rdn_base_bytes);
    init_fp16(reinterpret_cast<std::uint16_t*>(rdn_scales), rdn_rows * kD);
    init_u8(sgu_base, sgu_base_bytes);
    init_fp16(reinterpret_cast<std::uint16_t*>(sgu_scales), 2 * kI * kG);
    init_u8(sdn_base, sdn_base_bytes);
    init_fp16(reinterpret_cast<std::uint16_t*>(sdn_scales), kH * kD);
    init_router(router);
    init_bf16(x.data(), kH, 0.07f);
    init_bf16(dest.data(), kH, 0.2f);

    SparseMoe512x10Geometry geo;
    geo.n_experts = kE;
    geo.top_k = kK;
    geo.hidden = kH;
    geo.inter = kI;

    SparseMoe512x10CpuWeights w;
    w.router = router.data();
    w.routed_gate_up_codec = code;
    w.routed_gate_up_base = rgu_base;
    w.routed_gate_up_scales = reinterpret_cast<const std::uint16_t*>(rgu_scales);
    w.routed_down_codec = code;
    w.routed_down_base = rdn_base;
    w.routed_down_scales = reinterpret_cast<const std::uint16_t*>(rdn_scales);
    w.shared_gate_up_codec = code;
    w.shared_gate_up_base = sgu_base;
    w.shared_gate_up_scales = reinterpret_cast<const std::uint16_t*>(sgu_scales);
    w.shared_down_codec = code;
    w.shared_down_base = sdn_base;
    w.shared_down_scales = reinterpret_cast<const std::uint16_t*>(sdn_scales);

    // ---- warm call (fills the page cache, sanity check) -------------------------------
    {
        std::vector<std::int16_t> dest_copy(kH);
        std::memcpy(dest_copy.data(), dest.data(), kH * 2);
        auto t0 = std::chrono::steady_clock::now();
        sparse_moe_512x10_cpu(x.data(), 1, geo, w, dest_copy.data(), threads);
        auto t1 = std::chrono::steady_clock::now();
        bool ok = true;
        for (int i = 0; i < kH; ++i) {
            const std::uint16_t b = static_cast<std::uint16_t>(dest_copy[i]);
            if ((b & 0x7F80) == 0x7F80) { ok = false; break; }  // NaN
        }
        std::printf("warm call: %.3f ms (output %s)\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count(), ok ? "ok" : "NaN/Inf!");
        if (!ok) {
            std::printf("FAILED sanity\n");
            return 1;
        }
    }

    const std::int64_t flops = layer_flops(kK);
    std::vector<std::int16_t> dest_work(kH);
    auto seed_dest = [&]() -> std::int16_t* {
        std::memcpy(dest_work.data(), dest.data(), kH * 2);
        return dest_work.data();
    };

    // ---- steady-state timed runs (L2 flushed each run) --------------------------------
    {
        Stats s;
        for (int r = 0; r < iters; ++r) {
            std::int16_t* d = seed_dest();
            flush_l2(threads);
            auto t0 = std::chrono::steady_clock::now();
            sparse_moe_512x10_cpu(x.data(), 1, geo, w, d, threads);
            auto t1 = std::chrono::steady_clock::now();
            s.us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        print_run("steady-state (L2+L3 flushed)", s, s.median() * 1e-6, flops, accessed, threads);
    }

    // ---- second phase: same flush, more runs (memory-resident bank: no page-cache effect) ----
    if (cold) {
        Stats s;
        for (int r = 0; r < iters; ++r) {
            std::int16_t* d = seed_dest();
            flush_l2(threads);
            auto t0 = std::chrono::steady_clock::now();
            sparse_moe_512x10_cpu(x.data(), 1, geo, w, d, threads);
            auto t1 = std::chrono::steady_clock::now();
            s.us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        print_run("repeat (L2+L3 flushed)", s, s.median() * 1e-6, flops, accessed, threads);
    }

    // ---- summary line the plan's step-cost model consumes ------------------------------
    {
        std::int16_t* d = seed_dest();
        flush_l2(threads);
        auto t0 = std::chrono::steady_clock::now();
        sparse_moe_512x10_cpu(x.data(), 1, geo, w, d, threads);
        auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        std::printf("\nRESULT  layer time (T=1, whole 512-expert MoE layer) = %.3f ms  "
                    "-> FLOP/s %.2f GFLOP/s, expert weights %.2f GB/s (%zu MiB)\n",
                    sec * 1e3, flops / sec / 1e9, static_cast<double>(accessed) / sec / 1e9,
                    accessed >> 20);
    }

    std::free(rgu_base);
    std::free(rgu_scales);
    std::free(rdn_base);
    std::free(rdn_scales);
    std::free(sgu_base);
    std::free(sgu_scales);
    std::free(sdn_base);
    std::free(sdn_scales);
    return 0;
}
