"""Self-tests for tools/convert/common/encoder_imatrix.py.

Run:  python -m tools.convert.common.test_encoder_imatrix

Gates (nvfp.md §2.4):
  1. e4m3 scale words: bit parity vs the `gguf` package oracle
     (gguf.NVFP4.fp32_to_ue4m3) and the scalar ggml port, incl. edges.
  2. e4m3 decode: parity with the gguf oracle (x2 kvalues convention)
     and the standard decode used by the ninfer runtime codec.
  3. e2m1 snapping: parity vs the scalar ggml best_index_mxfp4 oracle,
     incl. exact tie points (lower-magnitude win).
  4. NVFP4 reference (band=None): byte parity, packed codes and scale
     words, vs the scalar ggml quantize_row_nvfp4_ref port.
  5. NVFP4 imatrix search: the weighted error is never worse than the
     reference; the chosen word stays in [ref, wmax].
  6. Q6/W8 reference: parity vs the MAXABS recipe (scalar port);
     imatrix search never worse than reference.
  7. Decode round-trip error bounds.
  8. weight_divisor / input_divisor sanity.
  9. Per-row bands: a [n, k] band must reproduce two independent
     [k]-band encodes of the row halves (fused expert objects).
"""

from __future__ import annotations

import struct

import numpy as np

from tools.convert.common import encoder_imatrix as enc


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_e4m3_words() -> None:
    from gguf import quants

    oracle = quants.NVFP4.fp32_to_ue4m3
    vals = np.concatenate(
        (
            np.array([0.0, 1e-10, 2 ** -9, 2 ** -9 * 7, 2 ** -9 * 7.4, 2 ** -6,
                      2 ** -6 * 1.5, 1.0, 223.999, 224.0, 224.001, 255.999,
                      256.0, 447.0, 448.0, 448.001, 1000.0], dtype=np.float32),
            np.abs(np.random.default_rng(0).standard_normal(200000))
            .astype(np.float32)
            * 300.0,
        )
    )
    mine = enc.fp32_to_e4m3_word(vals)
    ref = oracle(vals)
    diff = np.nonzero(mine != ref)[0]
    _check(diff.size == 0,
           f"e4m3 word parity vs gguf oracle: {diff.size} mismatches, "
           + (f"first at idx {diff[0]} (x={vals[diff[0]]}, mine={mine[diff[0]]}, "
              f"ref={ref[diff[0]]})" if diff.size else ""))
    # scalar ggml oracle spot check
    for x in vals[:17]:
        _check(enc.fp32_to_e4m3_word(np.array([x]))[0]
               == enc._oracle_fp32_to_ue4m3(float(x)),
               f"e4m3 scalar port mismatch at x={x}")
    print("  e4m3 words: OK (gguf oracle 200k random + 17 edges + scalar port)")


def test_e4m3_decode() -> None:
    from gguf import quants

    words = np.arange(127, dtype=np.uint8)
    mine = enc.e4m3_word_values(words)
    # gguf oracle uses the x0.5 convention; standard = 2x that
    gguf = quants.NVFP4.ue4m3_to_fp32(words) * 2.0
    diff = np.nonzero(mine != gguf)[0]
    _check(diff.size == 0,
           f"e4m3 decode parity: {diff.size} mismatches at words {diff}")
    _check(float(mine[126]) == 448.0, "word 126 must decode to 448")
    _check(float(mine[0]) == 0.0, "word 0 must decode to 0")
    _check(bool(np.all(np.diff(mine[1:]) > 0)), "word values must be strictly increasing")
    print("  e4m3 decode: OK (127 words vs gguf x2 oracle, monotone, max 448)")


def test_e2m1_snapping() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-7.0, 7.0, size=1_000_000).astype(np.float32)
    codes = enc.e2m1_codes(x)
    # Direct parity: e2m1_codes(x) minimizes |VAL[c] - x|; the oracle
    # minimizes |KVALUES[c]*d - x_arg| with KVALUES = 2*VAL, so
    # oracle(2x, 1.0) is the identical objective with identical ties.
    ok = 0
    for i in range(0, 1_000_000, 997):
        c_oracle = enc._oracle_best_index_mxfp4(float(2.0 * x[i]), 1.0)
        _check(c_oracle == int(codes[i]),
               f"e2m1 parity at x={x[i]}: mine={codes[i]} oracle={c_oracle}")
        ok += 1
    # exact tie points -> lower magnitude
    for tie, want in ((0.25, 0), (0.75, 1), (1.25, 2), (1.75, 3),
                      (2.5, 4), (3.5, 5), (5.0, 6)):
        _check(int(enc.e2m1_codes(np.array([np.float32(tie)]))[0]) == want,
               f"tie {tie} must go to magnitude index {want}")
        # negative tie: codes 0 and 8 both hold magnitude 0; ggml's
        # strict-< keeps the lowest index (0), so only non-zero
        # magnitudes map to want + 8.
        neg_code = want + 8 if want > 0 else 0
        _check(int(enc.e2m1_codes(np.array([np.float32(-tie)]))[0]) == neg_code,
               f"negative tie -{tie} must go to code {neg_code}")
    print(f"  e2m1 snapping: OK ({ok} oracle checks + 14 tie points)")


def test_nvfp4_reference_parity() -> None:
    rng = np.random.default_rng(2)
    t = (rng.standard_normal((96, 256)).astype(np.float32))
    t /= np.abs(t).max()
    t *= np.float32(2688.0)  # amax = 2688, like real t-space
    # sprinkle exact sub-block amax ties and zero blocks
    t[3, :16] = np.float32(2.0)
    t[5] = 0.0
    packed, words = enc.encode_nvfp4(t, None)
    for r in range(96):
        p_ref, w_ref = enc.oracle_quantize_row_nvfp4_ref(t[r])
        _check(np.array_equal(packed[r], p_ref),
               f"nvfp4 packed parity row {r}: "
               f"{packed[r][:8].tobytes().hex()} vs {p_ref[:8].tobytes().hex()}")
        _check(np.array_equal(words[r], w_ref),
               f"nvfp4 word parity row {r}: {words[r][:8]} vs {w_ref[:8]}")
    print("  nvfp4 reference: OK (96 rows x 16 sub-blocks byte parity vs ggml port)")


def test_nvfp4_imatrix_monotone() -> None:
    rng = np.random.default_rng(3)
    t = rng.standard_normal((512, 2560)).astype(np.float32)
    t /= np.abs(t).max()
    t *= np.float32(2688.0)
    band = np.abs(rng.standard_normal(2560)) ** 2
    band[(rng.random(2560) < 0.1)] *= 100.0  # some hot columns

    packed_ref, words_ref = enc.encode_nvfp4(t, None)
    packed_im, words_im = enc.encode_nvfp4(t, band)

    t16 = t.reshape(512, -1, 16)
    band16 = band.reshape(1, -1, 16)

    def weighted_err(packed, words):
        w = enc.decode_nvfp4(packed, words, 1.0)  # d_w=1 -> t-space values
        return float(((t16 - w.reshape(512, -1, 16)) ** 2 * band16).sum())

    e_ref = weighted_err(packed_ref, words_ref)
    e_im = weighted_err(packed_im, words_im)
    _check(e_im <= e_ref + 1e-6,
           f"imatrix must not be worse: ref={e_ref:.6g} im={e_im:.6g}")
    # chosen words must be >= ref words and legal
    _check(bool(np.all(words_im >= words_ref)), "words must be >= reference word")
    _check(bool(np.all(words_im <= 126)), "words must be legal (<= 126)")
    # reference (band=None) must still be reproducible
    packed_ref2, words_ref2 = enc.encode_nvfp4(t, None)
    _check(np.array_equal(packed_ref, packed_ref2), "reference must be deterministic")
    print(f"  nvfp4 imatrix: OK (err {e_ref:.6g} -> {e_im:.6g}, "
          f"{int((words_im != words_ref).sum())}/{words_im.size} sub-blocks changed)")


def test_row_split_reference_and_imatrix() -> None:
    # scalar oracle of the MAXABS reference recipe
    def oracle_int(w_row: np.ndarray, gs: int, qmax: int):
        codes = []
        scales = []
        for g0 in range(0, w_row.shape[0], gs):
            wg = w_row[g0:g0 + gs]
            amax = float(np.abs(wg).max())
            raw = np.float64(np.float32(amax)) / np.float64(qmax)
            s = np.float32(raw).astype(np.float16).astype(np.float32)
            if s == 0 and amax > 0:
                s = np.float32(2.0 ** -24)
            recip = np.float32(1.0 / np.float64(s))
            for j in range(gs):
                c = np.round(np.float32(wg[j] * recip))
                c = max(-qmax, min(qmax, float(c)))
                codes.append(int(c))
            scales.append(np.float16(s))
        return np.array(codes, np.int8), np.array(scales, np.float16)

    rng = np.random.default_rng(4)
    for gs, qmax in ((64, 31), (32, 127), (64, 127)):
        w = rng.standard_normal((48, 2560)).astype(np.float32)
        w[4, :64] = np.float32(1e-30)  # underflow-guard case (f16 -> 0)
        codes, scales = enc.encode_row_split(w, None, gs, qmax)
        for r in range(48):
            c_ref, s_ref = oracle_int(w[r], gs, qmax)
            _check(np.array_equal(codes[r].reshape(-1), c_ref),
                   f"int-group codes parity r={r} gs={gs} qmax={qmax}")
            _check(np.array_equal(scales[r], s_ref),
                   f"int-group scale parity r={r} gs={gs} qmax={qmax}")
        # imatrix never worse
        band = np.abs(rng.standard_normal(2560)) ** 2
        c_im, s_im = enc.encode_row_split(w, band, gs, qmax)

        def werr(c, s):
            dec = c.astype(np.float32) * s.astype(np.float32).reshape(48, -1, 1)
            diff = w - dec.reshape(48, -1)
            return float((diff**2 * band[None, :]).sum())

        e_ref = werr(codes, scales)
        e_im = werr(c_im, s_im)
        _check(e_im <= e_ref + 1e-6,
               f"int imatrix must not be worse gs={gs}: ref={e_ref:.6g} im={e_im:.6g}")
    print("  row-split q6/w8/q8: OK (96 rows x all shapes vs scalar MAXABS oracle)")


def test_decode_bound() -> None:
    rng = np.random.default_rng(5)
    w = rng.standard_normal((128, 1024)).astype(np.float32)
    a = float(np.abs(w).max())
    dw = struct.unpack("<f", enc.weight_divisor(w))[0]
    t = (w * dw).astype(np.float32)
    packed, words = enc.encode_nvfp4(t, None)
    dec = enc.decode_nvfp4(packed, words, dw)
    err = np.abs(w - dec).max()
    smax = float(enc.e4m3_word_values(words).max())
    bound = (0.5 * smax + 0.125 * smax * 6.0) / dw
    _check(err <= bound, f"decode err {err:.6g} exceeds bound {bound:.6g}")
    print(f"  decode bound: OK (max err {err:.6g} <= {bound:.6g})")


def test_divisors() -> None:
    rng = np.random.default_rng(6)
    w = rng.standard_normal((1024, 2560)).astype(np.float32) * 0.3
    a = float(np.abs(w).max())
    dw = struct.unpack("<f", enc.weight_divisor(w))[0]
    _check(abs(dw * a - 2688.0) / 2688.0 < 2e-6,
           f"d_w*amax must be ~2688: {dw * a}")
    band = np.abs(rng.standard_normal(2560)) ** 2 * 4.0
    r = float(np.sqrt(band).max())
    idiv = struct.unpack("<f", enc.input_divisor(band))[0]
    _check(abs(idiv * r - 2688.0) / 2688.0 < 2e-6,
           f"in_div*radius must be ~2688: {idiv * r}")
    # zero band -> finite default
    z = struct.unpack("<f", enc.input_divisor(np.zeros(8, np.float32)))[0]
    _check(np.isfinite(z), "zero band must give finite divisor")
    print("  divisors: OK (d_w, in_div round-trips; zero guard finite)")


def test_per_row_bands() -> None:
    rng = np.random.default_rng(20260830)
    # NVFP4: stacked per-row band == two independent per-half encodes
    for k in (640, 2560):
        g = (rng.standard_normal((128, k)) * 3.0).astype(np.float32)
        u = (rng.standard_normal((128, k)) * 2.0).astype(np.float32)
        g = (g * np.float32(1.5)).astype(np.float32)
        u = (u * np.float32(1.5)).astype(np.float32)
        t = np.concatenate([g, u], axis=0)
        bg = (rng.standard_normal(k) ** 2 * 7.0).astype(np.float32)
        bu = (rng.standard_normal(k) ** 2 * 3.0).astype(np.float32)
        pg_, wg_ = enc.encode_nvfp4(g, bg, row_chunk=64)
        pu_, wu_ = enc.encode_nvfp4(u, bu, row_chunk=64)
        band = np.concatenate([np.broadcast_to(bg, (128, k)),
                               np.broadcast_to(bu, (128, k))], axis=0)
        p, w = enc.encode_nvfp4(t, band, row_chunk=128)
        assert np.array_equal(w, np.concatenate([wg_, wu_], axis=0)), \
            "per-row NVFP4 words must equal the independent half encodes"
        assert np.array_equal(p, np.concatenate([pg_, pu_], axis=0)), \
            "per-row NVFP4 packed must equal the independent half encodes"
    # Q6: same property for the int-group encoder
    k, gs = 2560, 64
    wg = (rng.standard_normal((128, k)) * 0.05).astype(np.float32)
    wu = (rng.standard_normal((128, k)) * 0.4).astype(np.float32)
    w = np.concatenate([wg, wu], axis=0)
    bg = (rng.standard_normal(k) ** 2 * 9.0).astype(np.float32)
    bu = (rng.standard_normal(k) ** 2 * 2.0).astype(np.float32)
    cg, sg = enc.encode_row_split(wg, bg, gs, 31)
    cu, su = enc.encode_row_split(wu, bu, gs, 31)
    band = np.concatenate([np.broadcast_to(bg, (128, k)),
                           np.broadcast_to(bu, (128, k))], axis=0)
    c, s = enc.encode_row_split(w, band, gs, 31)
    assert np.array_equal(c, np.concatenate([cg, cu], axis=0)), \
        "per-row Q6 codes must equal the independent half encodes"
    assert np.array_equal(s, np.concatenate([sg, su], axis=0)), \
        "per-row Q6 scales must equal the independent half encodes"
    print("  per-row bands: OK (NVFP4 + Q6 == independent half encodes)")


def test_q4_asymmetric() -> None:
    # Q4: asymmetric codes [-8, 7], reference scale amax/7, no-band path
    def oracle_int(w_row: np.ndarray, gs: int, qmin: int, qmax: int):
        codes = []
        scales = []
        for g0 in range(0, w_row.shape[0], gs):
            wg = w_row[g0:g0 + gs]
            amax = float(np.abs(wg).max())
            raw = np.float64(np.float32(amax)) / np.float64(qmax)
            s = np.float32(raw).astype(np.float16).astype(np.float32)
            if s == 0 and amax > 0:
                s = np.float32(2.0 ** -24)
            recip = np.float32(1.0 / np.float64(s))
            for j in range(gs):
                c = max(qmin, min(qmax, float(np.round(np.float32(wg[j] * recip)))))
                codes.append(int(c))
            scales.append(np.float16(s))
        return np.array(codes, np.int8), np.array(scales, np.float16)

    rng = np.random.default_rng(7)
    # no-band path (MTP, D10) and banded path, gs=64, (qmin, qmax) = (-8, 7)
    w = rng.standard_normal((48, 2560)).astype(np.float32)
    w[3, :64] = np.float32(1e-30)  # underflow-guard case (f16 -> 0)
    codes, scales = enc.encode_row_split(w, None, gs=64, qmax=7, qmin=-8)
    _check(codes.shape == (48, 2560 // 64, 64), "q4 codes shape")
    _check(scales.shape == (48, 2560 // 64), "q4 scales shape")
    _check(int(codes.min()) >= -8 and int(codes.max()) <= 7,
           f"q4 code range [{int(codes.min())}, {int(codes.max())}] must fit [-8, 7]")
    for r in range(48):
        c_ref, s_ref = oracle_int(w[r], 64, -8, 7)
        _check(np.array_equal(codes[r].reshape(-1), c_ref),
               f"q4 no-band codes parity r={r}")
        _check(np.array_equal(scales[r], s_ref), f"q4 no-band scale parity r={r}")
    # banded path: per-group s_ref still divides by qmax=7, codes clipped to [-8, 7]
    band = np.abs(rng.standard_normal(2560)) ** 2
    c_im, s_im = enc.encode_row_split(w, band, gs=64, qmax=7, qmin=-8)
    _check(int(c_im.min()) >= -8 and int(c_im.max()) <= 7, "q4 banded code range")
    # default qmin stays symmetric (-qmax): Q6/W8 call sites are byte-unchanged
    c_sym, s_sym = enc.encode_row_split(w, None, gs=64, qmax=7)
    _check(np.array_equal(c_sym, codes), "qmin default must be -qmax")
    _check(np.array_equal(s_sym, scales), "qmin must not touch the scale")
    # decode round-trip: |w - dec| <= 0.5 * s per element (code steps are s)
    dec = codes.astype(np.float32) * scales.astype(np.float32)[:, :, None]
    err = np.abs(w - dec.reshape(48, -1)).max()
    smax = float(scales.max())
    _check(err <= 0.5 * smax + 1e-6, f"q4 decode err {err:.6g} > 0.5*smax {0.5 * smax:.6g}")
    print("  q4 asymmetric: OK (no-band + banded vs scalar oracle, range [-8, 7])")


def main() -> None:
    print("encoder_imatrix self-tests")
    test_e4m3_words()
    test_e4m3_decode()
    test_e2m1_snapping()
    test_nvfp4_reference_parity()
    test_nvfp4_imatrix_monotone()
    test_row_split_reference_and_imatrix()
    test_decode_bound()
    test_divisors()
    test_per_row_bands()
    test_q4_asymmetric()
    print("all self-tests passed")


if __name__ == "__main__":
    main()
