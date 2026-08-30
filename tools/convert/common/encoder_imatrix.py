"""Imatrix-aware BF16 -> JB-NVFP4 / Q6 / W8 numeric encoder (pure numpy, f32).

Contract: nvfp.md §2 (encoder spec). The objective is the MSE of
(importance x error): for a weight matrix W with imatrix importance band
b[j] (sum of squared input activations over calibration tokens, per input
column j), the encoder minimizes sum_j b[j] * (W - Q(W)) ** 2.

Numeric contracts (settled against the 27B NVFP4 artifact and the
llama.cpp reference algorithms):

NVFP4 (W = e2m1(c) * e4m3fn(s) / d_w, per 16-element sub-block):
  d_w = f32(2688.0 / A),  A = amax|W| over the whole object matrix.
        (2688 = 6 * 448: E2M1 max magnitude x E4M3FN max value; the 27B
        artifact's stored scale words reach 448 = word 126, which
        requires amax(t) = 2688, i.e. d_w = 2688/A.)
  t = W * d_w  (t-space; amax(t) = 2688)
  per 16-sub-block with a16 = amax|t|:
    reference scale word:  s_ref = RNE_e4m3(fn)(f32(a16 / 6.0))
        (bit-exact ggml_fp32_to_ue4m3 port, incl. its documented
        one-round-bit quirk)
    imatrix search: candidates = e4m3fn words (ascending value) in
        [s_ref, largest word with value <= a16]; pick argmin
        sum_j band[j] * (t_j - snap_j * s) ** 2, strict-< on ties
        (ascending order => the reference wins ties).
    codes: E2M1 nearest, ties -> lower magnitude
        (ggml best_index_mxfp4 strict-< convention).
  a16 == 0 -> word 0, all codes 0.

Q6 / W8 int-group (row-split-k128, gs in {64, 32}, w-space, no d_w):
  reference (bit-exact port of tools/convert/common/quantize.py):
    s_ref = f16(f32(f64(amax_g) / f64(qmax)));
    underflow guard: s_ref == 0 and amax_g > 0 -> s_ref = f16(2**-24)
    recip = f32(f64(1.0) / f64(s_ref))
    codes = clamp(round(w * recip), -qmax, qmax)   (round = RNE)
  imatrix search: 17 candidates s_i = RNE_f16(f32(s_ref * 2**(-i/4))),
    i = 0..16; for each candidate the recip is recomputed exactly per
    the reference recipe; pick argmin sum band * (w - c * s)**2,
    strict-< on ties (i=0 = reference wins ties).

Runtime decode (ninfer): weights W ~= e2m1(c) * e4m3(s) / d_w via
PTX cvt.rn (RNE) — the encoder's tie convention (lower magnitude) may
differ from the decoder's RNE at exact midpoints; that is the
documented, accepted behavior (measure-zero on real data).
"""

from __future__ import annotations

import struct

import numpy as np

# ---------------------------------------------------------------------------
# E2M1
# ---------------------------------------------------------------------------

# Standard E2M1 magnitudes (OCP NVFP4 code values), indices 0..7.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Midpoints between adjacent magnitudes (8 magnitudes -> 7 boundaries).
# x == midpoint ties to the lower magnitude (searchsorted side='left').
E2M1_MID = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def e2m1_codes(x: np.ndarray) -> np.ndarray:
    """Nearest E2M1 code in [0..15] for each element of x.

    x is the pre-divided value (t / scale). Ties go to the lower
    magnitude (ggml best_index_mxfp4 strict-< semantics). Note: the
    ggml table lists 0 at index 0 and again at index 8, so negative
    values that round to zero emit code 0 (+0), never 8 (-0).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    mag = np.searchsorted(E2M1_MID, np.abs(x), side="left").astype(np.uint8)
    sign = (x < 0) & (mag > 0)
    return (mag | (8 * sign.astype(np.uint8))).astype(np.uint8)


def e2m1_values(codes: np.ndarray) -> np.ndarray:
    """Standard E2M1 magnitude for each code (sign bit = bit 3)."""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    mag = E2M1_MAG[(codes & 7).astype(np.intp)].astype(np.float32)
    return np.where(codes >= 8, -mag, mag)


# ---------------------------------------------------------------------------
# E4M3FN scale words (unsigned, sign bit 0; word 0x7F is the reserved/NaN
# code and is never produced; word 0x7E == value 448 == max finite).
# ---------------------------------------------------------------------------

# Decoded value for each valid word 0..126, strictly increasing.
def _build_word_values() -> np.ndarray:
    # E4M3 word layout: 1 sign (0) + 4 exponent + 3 mantissa =>
    # word = (exp << 3) | man, man in 0..7 (word 0x7F is reserved/NaN).
    words = np.arange(127, dtype=np.uint8)
    exp = ((words >> 3) & 0xF).astype(np.float32)
    man = (words & 0x7).astype(np.float32)
    vals = np.where(
        exp == 0,
        man * np.float32(2.0 ** -9),
        (1.0 + man / 8.0) * np.float32(2.0) ** (exp - 7.0),
    )
    vals[126] = np.float32(448.0)  # word 0x7E: (1 + 6/8) * 2**8
    return vals.astype(np.float32)


E4M3_WORD_VALUES = _build_word_values()  # [127], strictly increasing, [0] = 0


def fp32_to_e4m3_word(x: np.ndarray) -> np.ndarray:
    """Vectorized bit-exact port of ggml_fp32_to_ue4m3 (ggml-impl.h:517).

    x must be >= 0. Quirk preserved: the normal path rounds using only
    the top 3 mantissa bits plus the 4th bit (not full RNE over all
    discarded bits). Values > 448 clamp to word 126 (448).
    """
    x = np.clip(np.ascontiguousarray(x, dtype=np.float32), 0.0, 448.0)
    bits = x.view(np.uint32)
    e32 = ((bits >> 23) & 0xFF).astype(np.int32) - 127
    m3 = ((bits >> 20) & 0x7).astype(np.int32)
    ue = e32 + 7

    # Subnormal branch (ue <= 0): word = man in 1..7, man = round(x * 512)
    sub_man = np.clip((x * np.float32(512.0) + np.float32(0.5)).astype(np.int32), 0, 7)
    sub_word = np.where(sub_man >= 1, sub_man, 0)

    # Normal branch: man3 + round_bit with carry to the exponent
    round_bit = ((bits >> 19) & 1).astype(np.int32)
    man = m3 + round_bit
    overflow = man > 7
    man2 = np.where(overflow, 0, man)
    ue2 = np.where(overflow, ue + 1, ue)
    normal_word = np.where(ue2 >= 15, 126, ((ue2 << 3) | man2))

    word = np.where(
        x <= 0.0,
        np.uint8(0),
        np.where(ue <= 0, sub_word, np.where(ue >= 15, 126, normal_word)),
    )
    return word.astype(np.uint8)


def e4m3_word_values(words: np.ndarray) -> np.ndarray:
    """Decode scale words (0..126) to their float values."""
    words = np.ascontiguousarray(words, dtype=np.uint8)
    if words.size:
        assert words.max() <= 126, "word 0x7F (NaN) must never appear"
    return E4M3_WORD_VALUES[words.astype(np.intp)].astype(np.float32)


def _candidate_words(a16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per sub-block: (candidate words [n, K], validity mask [n, K]).

    Candidates are ascending word values starting at the reference word
    RNE_e4m3(a16/6) up to the largest word with value <= a16. K is a
    fixed bound (24): the candidate range spans a factor of 6 in value
    = ~2.6 E4M3 octaves x 8 values <= 22 words, so 24 always suffices.
    """
    a16 = np.ascontiguousarray(a16, dtype=np.float32)
    shape = a16.shape
    a16f = a16.reshape(-1)
    n = a16f.size

    ref = fp32_to_e4m3_word(a16f / np.float32(6.0))  # [n] u8
    # largest word with value <= a16 (strictly increasing values):
    wmax = np.searchsorted(E4M3_WORD_VALUES, a16f, side="right").astype(np.int32) - 1
    wmax = np.clip(wmax, 0, 126)
    # a16 == 0: ref == 0, wmax == 0 -> single candidate word 0
    wmin = np.minimum(ref, wmax)
    count = wmax - wmin + 1  # [n], >= 1

    K = 24
    idx = np.arange(K, dtype=np.int32)[None, :]  # [1, K]
    words = (wmin[:, None] + idx)  # [n, K]
    valid = idx < count[:, None]
    # word 127 is reserved: it can never be a candidate (wmax <= 126)
    valid &= words <= 126
    return words.reshape(shape + (K,)), valid.reshape(shape + (K,))


# ---------------------------------------------------------------------------
# NVFP4 encode (t-space)
# ---------------------------------------------------------------------------


def weight_divisor(w: np.ndarray) -> bytes:
    """d_w = f32(2688 / amax|w|) little-endian (4 bytes)."""
    a = float(np.abs(w).max()) if w.size else 0.0
    if a == 0.0:
        return struct.pack("<f", np.float32(2688.0))
    return struct.pack("<f", np.float32(np.float32(2688.0) / np.float32(a)))


def input_divisor(band: np.ndarray) -> bytes:
    """Input (activation) divisor from an imatrix band.

    band[j] = sum over calibration tokens of x[t, j]**2  (per input
    column), so sqrt(band[j]) >= amax_t |x[t, j]| and
    R = max_j sqrt(band[j]) is a conservative activation radius.
    Returns f32(2688 / R) little-endian (4 bytes) — the 2688 factor
    mirrors the weight divisor (activation quantization is the same
    two-level E2M1 x E4M3 structure at runtime).
    """
    band = np.asarray(band, dtype=np.float32).ravel()
    r = float(np.sqrt(band).max()) if band.size else 0.0
    if r <= 0.0:
        return struct.pack("<f", np.float32(2688.0))
    return struct.pack("<f", np.float32(np.float32(2688.0) / np.float32(r)))


def encode_nvfp4(
    t: np.ndarray,
    band: np.ndarray | None = None,
    row_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a matrix already in t-space (t = W * d_w), amax(t) <= 2688.

    t: float32 [n, k], k % 16 == 0 (and in practice n % 128 == 0,
       k % 64 == 0 for the blockscale layout).
    band: float32 [k] imatrix importance per input column, or [n, k]
       per-row bands (fused expert objects: gate rows carry the gate
       band, up rows the up band, down rows the down band per expert),
       or None for the reference (no-imatrix) path.
    row_chunk: band-search row chunk (worker RAM bound; 256 keeps the
       per-candidate temporaries [chunk, k//16, 16] f32 ~2.6 MB at
       k = 2560).

    Returns (packed u8 [n, k//2] low nibble = even element,
             scale words u8 [n, k//16] in natural order).
    """
    t = np.ascontiguousarray(t, dtype=np.float32)
    n, k = t.shape
    assert k % 16 == 0, f"nvfp4 requires k % 16 == 0, got k={k}"

    t16 = t.reshape(n, k // 16, 16)
    a16 = np.abs(t16).max(axis=2)  # [n, k//16]

    if band is None:
        words = fp32_to_e4m3_word(a16 / np.float32(6.0))  # [n, k//16]
    else:
        band_a = np.asarray(band, dtype=np.float32)
        if band_a.ndim == 1:
            band16 = band_a.reshape(1, k // 16, 16)  # [1, g, 16], broadcast
        else:
            assert band_a.shape == (n, k), (
                f"per-row band must be [n, k]; got {band_a.shape} for n={n}, k={k}"
            )
            band16 = band_a.reshape(n, k // 16, 16)  # [n, g, 16]
        words = np.empty((n, k // 16), dtype=np.uint8)
        # Row-chunked, per-candidate search: the fused [rc, g, K, 16]
        # temporary is >1 GB per chunk at k = 2560 and blows worker RAM
        # at 32-way parallelism. Iterate the K candidates instead: the
        # per-candidate temporaries are [rc, g, 16] f32 (~2.6 MB per
        # 256-row chunk at k = 2560); the per-candidate 16-element error
        # sum and the first-wins argmin are bit-identical to the fused
        # formulation.
        for r0 in range(0, n, row_chunk):
            r1 = min(r0 + row_chunk, n)
            t16c = t16[r0:r1]  # [rc, g, 16]
            b16 = band16[r0:r1] if band16.shape[0] > 1 else band16
            cwords, cvalid = _candidate_words(a16[r0:r1])  # [rc, g, K]
            # invalid slots may hold words > 126: clamp before the lookup
            s = E4M3_WORD_VALUES[
                np.clip(cwords, 0, 126).astype(np.intp)
            ].astype(np.float32)  # [rc, g, K]
            # per-candidate summed error [rc, g, K]; invalid -> +inf
            err_sum = np.empty((r1 - r0, k // 16, cwords.shape[-1]),
                               dtype=np.float32)
            for cand in range(cwords.shape[-1]):
                sc = s[:, :, cand]  # [rc, g]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(sc[..., None] > 0, t16c / sc[..., None],
                                     0.0)
                v = e2m1_values(e2m1_codes(ratio)) * sc[..., None]
                err = ((t16c - v) ** 2) * b16  # [rc, g, 16]
                err_sum[:, :, cand] = np.where(
                    cvalid[:, :, cand], err.sum(axis=-1),
                    np.float32(np.inf),
                )
            # strict-< over ascending candidate order: first-wins argmin
            words[r0:r1] = np.take_along_axis(cwords, err_sum.argmin(
                axis=-1)[..., None], axis=-1)[..., 0]

    s_val = e4m3_word_values(words)  # [n, g]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_val[..., None] > 0, t16 / s_val[..., None], 0.0)
    codes = e2m1_codes(ratio)  # [n, g, 16]
    codes = codes.reshape(n, k)

    # Pack: low nibble = even index, high nibble = odd index.
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, words.astype(np.uint8)


def _first_argmin(words: np.ndarray, err: np.ndarray) -> np.ndarray:
    """First-index-wins argmin (strict <) over the candidate axis.

    words: [n, g, K] ascending; err: [n, g, K] per-candidate summed error.
    numpy argmin keeps the first occurrence on ties. Used by the tests
    (the fused path computes the same argmin inline).
    """
    return np.take_along_axis(words, err.argmin(axis=-1)[..., None],
                              axis=-1)[..., 0]


def decode_nvfp4(packed: np.ndarray, words: np.ndarray, d_w: float) -> np.ndarray:
    """Dequantize (test oracle): W ~= e2m1(c) * e4m3(s) / d_w.

    packed u8 [n, k//2] (low nibble = even element);
    words u8 [n, k//16] in natural order.
    """
    n, k2 = packed.shape
    lo = (packed & 0x0F).astype(np.uint8)
    hi = (packed >> 4).astype(np.uint8)
    codes = np.stack([lo, hi], axis=-1).reshape(n, k2 * 2)
    scale = e4m3_word_values(words).repeat(16, axis=1)  # [n, k]
    return e2m1_values(codes) * scale / d_w


# ---------------------------------------------------------------------------
# Int-group (Q6 / W8) encode
# ---------------------------------------------------------------------------


def _r16(x: np.ndarray) -> np.ndarray:
    """RNE cast to float16 and back (vectorized)."""
    return x.astype(np.float16).astype(np.float32)


def _int_scale_candidates(s_ref: np.ndarray) -> np.ndarray:
    """17 scale candidates s_i = RNE_f16(f32(s_ref * 2**(-i/4))), i=0..16.

    The factor 2**(-i/4) is formed in f64 then rounded to f32 (its exact
    f32 representation); the product is f32; the cast to f16 is RNE.
    Candidates that underflow to 0 are masked by the caller (+inf error;
    they are dominated by the i=0 reference).
    """
    s_ref = np.ascontiguousarray(s_ref, dtype=np.float32)
    i = np.arange(17, dtype=np.float64)
    factors = np.power(np.float64(2.0), -i / 4.0).astype(np.float32)  # [17]
    return _r16(s_ref[..., None] * factors[None, :])  # [n, 17]


def encode_row_split(
    w: np.ndarray,
    band: np.ndarray | None,
    gs: int,
    qmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode W [n, k] (k % gs == 0) with per-row int-group quantization.

    Returns (codes int8 [n, k//gs, gs], scales f16 [n, k//gs]).
    qmax: 31 (Q6) or 127 (W8).
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    n, k = w.shape
    assert k % gs == 0, f"row-split requires k % gs == 0, got k={k} gs={gs}"
    g = k // gs
    wg = w.reshape(n, g, gs)
    amax = np.abs(wg).max(axis=2)  # [n, g]
    if band is None:
        bandg = None
    else:
        band_a = np.asarray(band, dtype=np.float32)
        if band_a.ndim == 1:
            bandg = band_a.reshape(1, g, gs)  # [1, g, gs], broadcast
        else:
            assert band_a.shape == (n, k), (
                f"per-row band must be [n, k]; got {band_a.shape} for n={n}, k={k}"
            )
            bandg = band_a.reshape(n, g, gs)  # [n, g, gs]

    # Reference scale per the MAXABS recipe (bit-exact quantize.py port)
    raw = np.float64(amax.astype(np.float64)) / np.float64(qmax)
    s_ref = (raw.astype(np.float32)).astype(np.float16).astype(np.float32)
    # underflow guard
    s_ref = np.where((s_ref == 0) & (amax > 0), np.float32(2.0 ** -24), s_ref)

    if band is None:
        scales_f16 = s_ref.astype(np.float16)
        recip = np.zeros_like(s_ref)
        pos = s_ref > 0
        recip[pos] = np.float32(1.0 / s_ref[pos].astype(np.float64)).astype(
            np.float32
        )
        codes = np.clip(np.round(wg * recip[..., None]), -qmax, qmax).astype(np.int8)
        return codes, scales_f16

    # Row-chunked search: [rc, g, 17, gs] temporaries would exceed worker
    # RAM for full expert matrices at 32-way parallelism.
    cand_full = _int_scale_candidates(s_ref)  # [n, g, 17]
    codes_best = np.empty((n, g, gs), dtype=np.int8)
    scales_f16 = np.empty((n, g), dtype=np.float16)
    band_broadcast = bandg.shape[0] == 1  # 1-D band: one row for all rows
    for r0 in range(0, n, 256):
        r1 = min(r0 + 256, n)
        wg_c = wg[r0:r1]
        cand = cand_full[r0:r1]
        band_c = bandg if band_broadcast else bandg[r0:r1]
        # recip per candidate, exactly per the reference recipe (0 scales
        # guarded so the code values are finite; masked by +inf error below)
        safe_cand = np.where(cand > 0, cand, np.float32(1.0))
        recip = np.float32(1.0 / safe_cand.astype(np.float64)).astype(np.float32)
        codes_all = np.clip(
            np.round(wg_c[..., None, :] * recip[..., :, None]), -qmax, qmax
        )
        err = ((wg_c[..., None, :] - codes_all.astype(np.float32) * cand[..., :, None]) ** 2)
        err = err * band_c[..., None, :]
        err = np.where(cand[..., :, None] > 0, err, np.float32(np.inf))
        best = err.sum(axis=-1).argmin(axis=-1)  # first-wins on ties
        scales_f16[r0:r1] = (
            np.take_along_axis(cand, best[..., None], axis=-1)[..., 0]
        ).astype(np.float16)
        recip_best = np.take_along_axis(recip, best[..., None], axis=-1)[..., 0]
        codes_best[r0:r1] = np.clip(
            np.round(wg_c * recip_best[..., None]), -qmax, qmax
        ).astype(np.int8)
    return codes_best, scales_f16


# ---------------------------------------------------------------------------
# Self-contained oracles (reference algorithms ported from llama.cpp,
# scalar loops — used only by the self-tests, never by the encoder)
# ---------------------------------------------------------------------------


def _oracle_fp32_to_ue4m3(x: float) -> int:
    """Scalar ggml_fp32_to_ue4m3."""
    if not (x > 0.0):
        return 0
    if x > 448.0:
        x = 448.0
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    e32 = ((b >> 23) & 0xFF) - 127
    m3 = (b >> 20) & 0x7
    ue = e32 + 7
    if ue <= 0:
        man = int(x * 512.0 + 0.5)
        if man > 7:
            man = 7
        if man < 1:
            return 0
        return man
    if ue >= 15:
        return 0x7E
    man = m3 + ((b >> 19) & 1)
    if man > 7:
        man = 0
        ue += 1
        if ue >= 15:
            return 0x7E
    return (ue << 3) | man


def _oracle_ue4m3_to_fp32(word: int) -> float:
    """Scalar ggml_ue4m3_to_fp32 (note: x0.5 convention, paired with the
    x2 kvalues table in the ggml reference)."""
    if word == 0 or word == 0x7F:
        return 0.0
    exp = (word >> 3) & 0xF
    man = word & 0x7
    if exp == 0:
        raw = man * 2.0 ** -9
    else:
        raw = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return raw * 0.5


_KVALUES = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8
)


def _oracle_best_index_mxfp4(x: float, d: float) -> int:
    """Scalar ggml best_index_mxfp4 (strict <, ties -> lowest index)."""
    best_index = 0
    best_err = abs(float(_KVALUES[0]) * d - x)
    for i in range(1, 16):
        err = abs(float(_KVALUES[i]) * d - x)
        if err < best_err:
            best_index = i
            best_err = err
    return best_index


def oracle_quantize_row_nvfp4_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scalar port of ggml quantize_row_nvfp4_ref.

    x: f32 [k], k % 16 == 0 (works per 16-sub-block, ggml QK = 64 is 4
    independent sub-blocks with identical per-sub-block logic).
    Returns (packed u8 [k//2] low nibble = even, scale words u8 [k//16]).
    NOTE: this oracles the *scale/code words* only; it has no d_w (ggml
    blocks are self-contained). Feed t = W * d_w to compare against
    encode_nvfp4(t, band=None).
    """
    k = x.shape[0]
    assert k % 16 == 0
    words = np.empty(k // 16, dtype=np.uint8)
    codes = np.empty(k, dtype=np.uint8)
    for s in range(k // 16):
        xb = x[s * 16 : (s + 1) * 16]
        amax = float(np.abs(xb).max())
        # ggml divides in f32: amax / 6.0f (single rounding, not f64)
        w = _oracle_fp32_to_ue4m3(float(np.float32(amax) / np.float32(6.0)))
        words[s] = w
        d = _oracle_ue4m3_to_fp32(w)
        for j in range(16):
            c = _oracle_best_index_mxfp4(float(xb[j]), d)
            codes[s * 16 + j] = c & 0x0F
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    return packed, words

# ---------------------------------------------------------------------------
# FP8 row-scale (E4M3FN codes + BF16 scale)
#
# Reference profile MAXABS_BF16S_RECIP_E4M3FN_RNE_V1, bit-exact with
# tools/convert/qwen3_8_27b/fp8_embedding.py. The imatrix path searches
# 17 per-row scale candidates (mirroring the int-group convention, but
# BF16-rounded) and minimizes sum_j band[j] * (x - snap_e4m3(x/s) * s)^2;
# candidate 0 is the reference scale, so the result is never worse.
# ---------------------------------------------------------------------------

_E4M3FN_MAX = np.float32(448.0)
_BF16_MIN_SUBNORMAL_WORD = np.uint16(0x0001)


def _bf16_rne_words(values: np.ndarray) -> np.ndarray:
    """RNE cast of nonnegative binary32 values to exact BF16 words (u16)."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    upper = bits >> np.uint32(16)
    rounded = (
        bits.astype(np.uint64)
        + np.uint64(0x7FFF)
        + (upper & np.uint32(1)).astype(np.uint64)
    ) >> np.uint64(16)
    return rounded.astype(np.uint16)


def _bf16_words_to_float32(words: np.ndarray) -> np.ndarray:
    """Exact BF16 u16 words to binary32."""
    words = np.ascontiguousarray(words, dtype=np.uint16)
    return (words.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _round_e4m3fn_rne(values: np.ndarray) -> np.ndarray:
    """Round bounded finite binary32 values to exact signed E4M3FN words (u8).

    Ties go to the even word index (standard RNE over the OCP E4M3FN set,
    0x7F reserved as NaN so the magnitude index is 0..0x7E).
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or np.any(np.abs(values) > _E4M3FN_MAX):
        raise ValueError("E4M3FN rounding input must be finite and bounded")
    magnitude = np.abs(values)
    upper = np.searchsorted(E4M3_WORD_VALUES, magnitude, side="left").astype(np.int16)
    upper = np.minimum(upper, 0x7E)
    lower = np.maximum(upper - 1, 0)
    lower_distance = magnitude - E4M3_WORD_VALUES[lower]
    upper_distance = E4M3_WORD_VALUES[upper] - magnitude
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    words = np.where(choose_upper, upper, lower).astype(np.uint8)
    words = words | np.where(np.signbit(values), np.uint8(0x80), np.uint8(0))
    return words


def _e4m3fn_signed_values(words: np.ndarray) -> np.ndarray:
    """Decode signed E4M3FN code words (u8) to binary32 values.

    The magnitude index is words & 0x7F (0..126 for any word these
    quantizers emit); word 0x7F (NaN) is never produced.
    """
    words = np.ascontiguousarray(words, dtype=np.uint8)
    mag = E4M3_WORD_VALUES[(words & np.uint8(0x7F)).astype(np.intp)].astype(np.float32)
    sign = np.where(words & np.uint8(0x80), np.float32(-1.0), np.float32(1.0))
    return sign * mag


def quantize_fp8_row_scaled(
    w: np.ndarray,
    band: np.ndarray | None = None,
    row_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a binary32 matrix to row-scaled FP8 (E4M3FN codes + BF16 scale).

    w: float32 [n, k] (finite).
    band: float32 [k] per-input-column importance (broadcast over rows),
          or [n, k] per-row bands, or None for the reference
          MAXABS_BF16S_RECIP_E4M3FN_RNE_V1 profile (bit-exact with
          tools/convert/qwen3_8_27b/fp8_embedding.py).
    row_chunk: band-search row chunk (worker RAM bound; the f64
          temporaries are [chunk, k]).

    Returns (codes u8 [n, k], scale_words u16 [n]) where scale_words are
    exact nonnegative BF16 words (sign bit 0). Pack with
    tools.artifact.layouts.encode_fp8_row_scaled.

    Objective: argmin over per-row scale s of sum_j band[j] * (x - q)^2
    with q = snap_e4m3fn_rne(clip(x/s, +-448)) * s. Candidate 0 is the
    reference maxabs scale, so the imatrix result is never worse.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"fp8 row-scale requires rank-2 input, got {w.ndim}d")
    n, k = w.shape
    if not np.isfinite(w).all():
        raise ValueError("fp8 source contains NaN or infinity")

    amax = np.abs(w).max(axis=1)  # [n]
    zero_rows = amax == np.float32(0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (
            amax.astype(np.float64) / np.float64(_E4M3FN_MAX)
        ).astype(np.float32)
    s_ref_words = _bf16_rne_words(raw_scale)
    underflow = (s_ref_words == np.uint16(0)) & ~zero_rows
    if underflow.any():
        s_ref_words = s_ref_words.copy()
        s_ref_words[underflow] = _BF16_MIN_SUBNORMAL_WORD
    s_ref32 = _bf16_words_to_float32(s_ref_words)  # [n] f32

    if band is None:
        # Reference path, bit-exact vs fp8_embedding._quantize_host_rows.
        recip = np.zeros(n, dtype=np.float32)
        nz = ~zero_rows
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            recip[nz] = np.float32(
                1.0 / s_ref32[nz].astype(np.float64)
            ).astype(np.float32)
            normalized = (w.astype(np.float64) * recip[:, None].astype(np.float64)).astype(
                np.float32
            )
        normalized = np.clip(normalized, -_E4M3FN_MAX, _E4M3FN_MAX)
        codes = _round_e4m3fn_rne(normalized)
        codes = np.where(zero_rows[:, None], np.uint8(0), codes).astype(np.uint8)
        return codes, s_ref_words

    # Imatrix path: per-row band-weighted scale search.
    band_a = np.asarray(band, dtype=np.float32)
    if band_a.ndim == 1:
        if band_a.shape[0] != k:
            raise ValueError(f"band must be [k], got {band_a.shape} for k={k}")
        band_b = band_a.reshape(1, k)
        band_per_row = False
    else:
        if band_a.shape != (n, k):
            raise ValueError(f"per-row band must be [n, k], got {band_a.shape}")
        band_b = band_a
        band_per_row = True

    j = np.arange(17, dtype=np.float64)
    factors = np.power(np.float64(2.0), -j / 4.0).astype(np.float32)  # [17]
    cand_words = _bf16_rne_words(s_ref32[:, None] * factors[None, :])  # [n,17] u16
    cand32 = _bf16_words_to_float32(cand_words)  # [n,17] f32

    codes_out = np.empty((n, k), dtype=np.uint8)
    scale_out = np.empty(n, dtype=np.uint16)
    for r0 in range(0, n, row_chunk):
        r1 = min(r0 + row_chunk, n)
        wc = w[r0:r1]
        bc = band_b if not band_per_row else band_b[r0:r1]
        cw = cand_words[r0:r1]  # [rc,17] u16
        c32 = cand32[r0:r1]  # [rc,17] f32
        rc = r1 - r0
        wc64 = wc.astype(np.float64)  # reuse across the 17 candidates
        err = np.empty((rc, 17), dtype=np.float32)
        for c in range(17):
            s = c32[:, c]  # [rc]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                recip = np.where(
                    s > 0,
                    np.float32(1.0 / s.astype(np.float64)).astype(np.float32),
                    np.float32(0.0),
                )
                normalized = (wc64 * recip[:, None].astype(np.float64)).astype(np.float32)
            normalized = np.clip(normalized, -_E4M3FN_MAX, _E4M3FN_MAX)
            deq = _e4m3fn_signed_values(_round_e4m3fn_rne(normalized)) * s[:, None]
            diff = wc - deq
            err_c = (diff * diff) * bc
            err[:, c] = err_c.sum(axis=1).astype(np.float32)
            bad = s == np.float32(0.0)
            if bad.any():
                err[bad, c] = np.float32(np.inf)
        best = err.argmin(axis=1)  # first-wins on ties
        idx = np.arange(rc)
        s_best = c32[idx, best]  # [rc]
        cw_best = cw[idx, best]  # [rc] u16
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            recip = np.where(
                s_best > 0,
                np.float32(1.0 / s_best.astype(np.float64)).astype(np.float32),
                np.float32(0.0),
            )
            normalized = (wc64 * recip[:, None].astype(np.float64)).astype(np.float32)
        normalized = np.clip(normalized, -_E4M3FN_MAX, _E4M3FN_MAX)
        codes_out[r0:r1] = _round_e4m3fn_rne(normalized)
        codes_out[r0:r1] = np.where(
            zero_rows[r0:r1][:, None], np.uint8(0), codes_out[r0:r1]
        ).astype(np.uint8)
        scale_out[r0:r1] = cw_best
    return codes_out, scale_out
